#!/bin/bash
# Evaluate cells as they finish training, in two rounds: matched + clean-eval.
# Re-evaluates only cells whose .done exists but no prediction yet.
set -e
ROOT=/data/dual_noise
SEED=${1:-42}
LEVELS=(0 1 2 3 4 5)
QUALITY=(0.0 0.2 0.4 0.6 0.8 1.0)

cd $ROOT
source /data/anaconda3/etc/profile.d/conda.sh
conda activate dualnoise

RESULTS=$ROOT/experiments/results.csv
CLEAN_LS=$ROOT/data/processed/ccks_ls/L0_q1.0.json

# Use a single GPU (eval uses very little memory; training is on GPUs 0-7)
# Pick GPU 7 since it tends to free up first when batches are uneven
export CUDA_VISIBLE_DEVICES=7

n_eval=0
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cfg=$ROOT/experiments/logs/cfg_L${lt}_q${q}_seed${SEED}.json
    done_file=$ROOT/experiments/logs/train_L${lt}_q${q}_seed${SEED}.log.done
    best_model=$ROOT/runs/ccks_L${lt}_q${q}_seed${SEED}/best/model.safetensors
    if [ ! -f "$cfg" ] || [ ! -f "$done_file" ] || [ ! -f "$best_model" ]; then
      continue
    fi

    # Matched eval
    matched_pred=$ROOT/experiments/predictions/encoder_crf_zh_L${lt}_q${q}_le${lt}_seed${SEED}.json
    matched_ls=$ROOT/data/processed/ccks_ls/L${lt}_q${q}.json
    if [ ! -f "$matched_pred" ]; then
      python code/eval_cell.py --config "$cfg" --eval-ls "$matched_ls" \
        --out "$matched_pred" --result-row "$RESULTS" \
        --baseline encoder_crf --lang zh --lt $lt --rar $q --le $lt --seed $SEED 2>&1 | tail -1
      n_eval=$((n_eval + 1))
    fi

    # Clean eval (this baseline's model evaluated on L0_q1.0)
    clean_pred=$ROOT/experiments/predictions/encoder_crf_zh_L${lt}_q${q}_le0_seed${SEED}.json
    if [ ! -f "$clean_pred" ]; then
      python code/eval_cell.py --config "$cfg" --eval-ls "$CLEAN_LS" \
        --out "$clean_pred" --result-row "$RESULTS" \
        --baseline encoder_crf --lang zh --lt $lt --rar $q --le 0 --seed $SEED 2>&1 | tail -1
      n_eval=$((n_eval + 1))
    fi
  done
done

echo "[$(date +%H:%M:%S)] eval rounds run: $n_eval"
echo "results: $(wc -l < $RESULTS) rows"
tail -3 $RESULTS
