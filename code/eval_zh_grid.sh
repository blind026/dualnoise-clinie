#!/bin/bash
# Evaluate all 36 ZH cells under matched-noise protocol (eval cell == train cell).
# Also evaluates each model on the L0_q1.0 (clean) cell to give a clean-eval column.
set -e
ROOT=/data/dual_noise
SEED=${1:-42}
LEVELS=(0 1 2 3 4 5)
QUALITY=(0.0 0.2 0.4 0.6 0.8 1.0)

cd $ROOT
source /data/anaconda3/etc/profile.d/conda.sh
conda activate dualnoise

# Clear stale results CSV (we'll re-write)
RESULTS=$ROOT/experiments/results.csv
[ -f "$RESULTS" ] && mv "$RESULTS" "${RESULTS}.bak.$(date +%H%M%S)"

CLEAN_LS=$ROOT/data/processed/ccks_ls/L0_q1.0.json   # used as clean-eval surface

for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cfg=$ROOT/experiments/logs/cfg_L${lt}_q${q}_seed${SEED}.json
    if [ ! -f "$cfg" ]; then continue; fi
    if [ ! -f "$ROOT/runs/ccks_L${lt}_q${q}_seed${SEED}/best/model.safetensors" ]; then
      echo "SKIP L${lt}_q${q}: no best model"
      continue
    fi

    # Matched: evaluate on the same noise level (eval LS = same lt)
    matched_ls=$ROOT/data/processed/ccks_ls/L${lt}_q1.0.json   # take cell with senior labels at lt as "noisy clean test"
    pred_matched=$ROOT/experiments/predictions/encoder_crf_zh_L${lt}_q${q}_le${lt}_seed${SEED}.json
    python code/eval_cell.py --config $cfg --eval-ls $matched_ls \
      --out $pred_matched --result-row $RESULTS \
      --baseline encoder_crf --lang zh --lt $lt --rar $q --le $lt --seed $SEED 2>&1 | tail -1

    # Clean: evaluate on L0_q1.0
    pred_clean=$ROOT/experiments/predictions/encoder_crf_zh_L${lt}_q${q}_le0_seed${SEED}.json
    python code/eval_cell.py --config $cfg --eval-ls $CLEAN_LS \
      --out $pred_clean --result-row $RESULTS \
      --baseline encoder_crf --lang zh --lt $lt --rar $q --le 0 --seed $SEED 2>&1 | tail -1
  done
done

echo "==================="
echo "Results CSV: $RESULTS"
wc -l $RESULTS
head -1 $RESULTS
echo "..."
tail -5 $RESULTS
