#!/bin/bash
# DualNoise-ClinIE — Encoder-CRF full grid runner
#
# 1. Convert all 36 ZH cells to Label Studio JSON
# 2. Run BERT/prepare_data.py for each
# 3. Fan out 36 training jobs across 8 GPUs (4-5 jobs per GPU)
#
# Usage:  bash code/run_zh_grid.sh [num_epochs] [seed]
#
set -e
ROOT=/data/dual_noise
NUM_EPOCHS=${1:-4}
SEED=${2:-42}
GPUS=8
JOBS_PER_GPU=1   # mBERT base is ~700MB, can pack but be conservative

LEVELS=(0 1 2 3 4 5)
QUALITY=(0.0 0.2 0.4 0.6 0.8 1.0)

cd $ROOT
source /data/anaconda3/etc/profile.d/conda.sh
conda activate dualnoise

mkdir -p data/processed/ccks_ls data/processed/ccks_prepared experiments/logs runs

echo "[$(date +%H:%M:%S)] Step 1: Convert 36 cells to Label Studio format"
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cell="data/processed/ccks_cells/L${lt}_q${q}.jsonl"
    out="data/processed/ccks_ls/L${lt}_q${q}.json"
    if [ ! -f "$out" ]; then
      python code/convert_to_labelstudio.py --input "$cell" --output "$out" --lang zh --schema flat 2>&1 | tail -3 | sed "s/^/[L${lt}_q${q}] /"
    fi
  done
done

echo "[$(date +%H:%M:%S)] Step 2: prepare_data per cell"
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    ls_file="data/processed/ccks_ls/L${lt}_q${q}.json"
    out_dir="data/processed/ccks_prepared/L${lt}_q${q}"
    if [ ! -f "$out_dir/train.json" ]; then
      python BERT/pre_struct/kv_ner/prepare_data.py \
        --config code/kv_ner_config_dualnoise.json \
        --input "$ls_file" \
        --output_dir "$out_dir" 2>&1 | tail -2 | sed "s/^/[prep L${lt}_q${q}] /"
    fi
  done
done

echo "[$(date +%H:%M:%S)] Step 3: Launch 36 training jobs across $GPUS GPUs"

# Build training config per cell, then dispatch to GPU via tmux
gpu_idx=0
job_idx=0
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    prepared_dir="data/processed/ccks_prepared/L${lt}_q${q}"
    run_dir="runs/ccks_L${lt}_q${q}_seed${SEED}"
    cfg_file="experiments/logs/cfg_L${lt}_q${q}_seed${SEED}.json"
    log_file="experiments/logs/train_L${lt}_q${q}_seed${SEED}.log"

    # Build per-cell config: override paths and epochs
    python3 - <<PY
import json
cfg = json.load(open('code/kv_ner_config_dualnoise.json'))
cfg['train']['data_path'] = '${ROOT}/${prepared_dir}/train.json'
cfg['train']['val_data_path'] = '${ROOT}/${prepared_dir}/val.json'
cfg['train']['output_dir'] = '${ROOT}/${run_dir}'
cfg['train']['num_train_epochs'] = ${NUM_EPOCHS}
cfg['train']['seed'] = ${SEED}
json.dump(cfg, open('${cfg_file}', 'w'), ensure_ascii=False, indent=2)
PY

    gpu=$((job_idx % GPUS))
    qsafe=$(echo "$q" | tr -d '.')
    sess="tr_L${lt}_q${qsafe}"
    echo "  [$(date +%H:%M:%S)] dispatch $sess -> GPU $gpu"

    # Avoid clobbering existing tmux session
    tmux kill-session -t "$sess" 2>/dev/null || true
    tmux new-session -d -s "$sess" \
      "cd $ROOT/BERT && CUDA_VISIBLE_DEVICES=$gpu python pre_struct/kv_ner/train.py --config $ROOT/$cfg_file 2>&1 | tee $ROOT/$log_file; echo DONE > $ROOT/$log_file.done"

    job_idx=$((job_idx + 1))
    # If we filled GPUS×JOBS_PER_GPU concurrent slots, wait for one batch to finish
    if [ $((job_idx % (GPUS * JOBS_PER_GPU))) -eq 0 ]; then
      echo "  [$(date +%H:%M:%S)] launched batch of $((GPUS * JOBS_PER_GPU)); waiting for completion before next batch..."
      while true; do
        done_count=$(ls $ROOT/experiments/logs/*.done 2>/dev/null | wc -l)
        if [ "$done_count" -ge "$job_idx" ]; then break; fi
        sleep 30
      done
      echo "  [$(date +%H:%M:%S)] batch done ($done_count/$job_idx). Continuing."
    fi
  done
done

# Wait for the last batch
echo "[$(date +%H:%M:%S)] Waiting for last batch..."
while true; do
  done_count=$(ls $ROOT/experiments/logs/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -ge "$job_idx" ]; then break; fi
  sleep 30
done
echo "[$(date +%H:%M:%S)] All $job_idx jobs done."
