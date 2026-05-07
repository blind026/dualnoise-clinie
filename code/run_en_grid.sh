#!/bin/bash
# DualNoise-ClinIE — English encoder-CRF grid runner
# - 36 cells (6 noise × 6 quality) on MTSamples 1500-doc subset
# - Uses use_bilstm=False from the start (avoids the crash bug found in ZH grid)
set -e
ROOT=/data/dual_noise
NUM_EPOCHS=${1:-4}
SEED=${2:-42}
GPUS=8
JOBS_PER_GPU=1

LEVELS=(0 1 2 3 4 5)
QUALITY=(0.0 0.2 0.4 0.6 0.8 1.0)

cd $ROOT
source /data/anaconda3/etc/profile.d/conda.sh
conda activate dualnoise

mkdir -p data/processed/mtsamples_cells data/processed/mtsamples_ls data/processed/mtsamples_prepared experiments/logs runs

echo "[$(date +%H:%M:%S)] Step 0: Annotation simulator on 36 cells"
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cell_in="data/processed/mtsamples_noise/L${lt}.jsonl"
    cell_out="data/processed/mtsamples_cells/L${lt}_q${q}.jsonl"
    if [ ! -f "$cell_out" ] && [ -f "$cell_in" ]; then
      python code/annotation_error_sim.py --input "$cell_in" --output "$cell_out" --rho-a-r $q --seed $SEED 2>&1 | tail -1 | sed "s/^/[L${lt}_q${q}] /"
    fi
  done
done

echo "[$(date +%H:%M:%S)] Step 1: Convert 36 cells to Label Studio (kv schema, EN keeps section headers)"
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cell="data/processed/mtsamples_cells/L${lt}_q${q}.jsonl"
    out="data/processed/mtsamples_ls/L${lt}_q${q}.json"
    if [ ! -f "$out" ] && [ -f "$cell" ]; then
      python code/convert_to_labelstudio.py --input "$cell" --output "$out" --lang en --schema flat 2>&1 | tail -3 | sed "s/^/[ls L${lt}_q${q}] /"
    fi
  done
done

# Build EN config (different from ZH: different label_map for the silver labels)
cat > $ROOT/code/kv_ner_config_en.json <<EOF
{
  "model_name_or_path": "/data/dual_noise/models/mbert",
  "tokenizer_name_or_path": "/data/dual_noise/models/mbert",
  "label_map": {
    "chief_complaint": "CC",
    "history_of_present_illness": "HPI",
    "past_medical_history": "PMH",
    "social_family_history": "SFH",
    "medications_allergies": "MED",
    "physical_examination": "PE",
    "diagnostic_studies": "LAB",
    "assessment_diagnosis": "DX",
    "plan_treatment": "PLAN"
  },
  "max_seq_length": 512,
  "label_all_tokens": true,
  "chunk_size": 500,
  "chunk_overlap": 50,
  "merge_adjacent_gap": 5,
  "value_attach_window": 60,
  "value_same_line_only": true,
  "data": {"output_dir": "data/mtsamples_prepared", "train_ratio": 0.9, "seed": 42, "input_files": []},
  "train": {
    "data_path": "",
    "val_data_path": "",
    "test_split_ratio": 0.5,
    "output_dir": "",
    "train_batch_size": 16, "eval_batch_size": 16,
    "num_workers": 0, "pin_memory": false,
    "learning_rate": 2e-5, "weight_decay": 0.03,
    "num_train_epochs": ${NUM_EPOCHS},
    "gradient_accumulation_steps": 1, "warmup_ratio": 0.05,
    "max_grad_norm": 1.0, "dropout": 0.2,
    "freeze_encoder": false, "unfreeze_last_n_layers": null,
    "use_bilstm": false, "use_conv": false,
    "encoder_learning_rate": 2e-5, "head_learning_rate": 1e-4,
    "use_amp": true, "amp_dtype": "bf16",
    "grad_checkpointing": false,
    "seed": ${SEED}
  },
  "predict": {"input_path": "", "output_path": "", "model_dir": "", "batch_size": 16}
}
EOF

echo "[$(date +%H:%M:%S)] Step 2: prepare_data per cell"
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    ls_file="data/processed/mtsamples_ls/L${lt}_q${q}.json"
    out_dir="data/processed/mtsamples_prepared/L${lt}_q${q}"
    if [ ! -f "$out_dir/train.json" ] && [ -f "$ls_file" ]; then
      python BERT/pre_struct/kv_ner/prepare_data.py \
        --config $ROOT/code/kv_ner_config_en.json \
        --input "$ls_file" \
        --output_dir "$out_dir" 2>&1 | tail -1 | sed "s/^/[prep L${lt}_q${q}] /"
    fi
  done
done

echo "[$(date +%H:%M:%S)] Step 3: Launch 36 training jobs (no BiLSTM) across $GPUS GPUs"
gpu_idx=0
job_idx=0
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    prepared_dir="data/processed/mtsamples_prepared/L${lt}_q${q}"
    run_dir="runs/mtsamples_L${lt}_q${q}_seed${SEED}"
    cfg_file="experiments/logs/cfg_en_L${lt}_q${q}_seed${SEED}.json"
    log_file="experiments/logs/train_en_L${lt}_q${q}_seed${SEED}.log"
    [ ! -f "$ROOT/$prepared_dir/train.json" ] && continue

    python3 - <<PY
import json
cfg = json.load(open('$ROOT/code/kv_ner_config_en.json'))
cfg['train']['data_path'] = '$ROOT/$prepared_dir/train.json'
cfg['train']['val_data_path'] = '$ROOT/$prepared_dir/val.json'
cfg['train']['output_dir'] = '$ROOT/$run_dir'
json.dump(cfg, open('$cfg_file','w'), ensure_ascii=False, indent=2)
PY

    gpu=$((job_idx % GPUS))
    qsafe=$(echo "$q" | tr -d '.')
    sess="en_L${lt}_q${qsafe}"
    tmux kill-session -t "$sess" 2>/dev/null || true
    tmux new-session -d -s "$sess" \
      "source /data/anaconda3/etc/profile.d/conda.sh && conda activate dualnoise && cd $ROOT/BERT && CUDA_VISIBLE_DEVICES=$gpu python pre_struct/kv_ner/train.py --config $ROOT/$cfg_file 2>&1 | tee $ROOT/$log_file; echo DONE > $ROOT/$log_file.done"
    job_idx=$((job_idx + 1))
    if [ $((job_idx % (GPUS * JOBS_PER_GPU))) -eq 0 ]; then
      echo "  [$(date +%H:%M:%S)] launched batch; waiting for $job_idx done..."
      while true; do
        c=$(ls $ROOT/experiments/logs/train_en_*.log.done 2>/dev/null | wc -l)
        if [ "$c" -ge "$job_idx" ]; then break; fi
        sleep 30
      done
      echo "  [$(date +%H:%M:%S)] batch done"
    fi
  done
done

echo "[$(date +%H:%M:%S)] Waiting for last EN batch..."
while true; do
  c=$(ls $ROOT/experiments/logs/train_en_*.log.done 2>/dev/null | wc -l)
  if [ "$c" -ge "$job_idx" ]; then break; fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] All $job_idx EN jobs done."
