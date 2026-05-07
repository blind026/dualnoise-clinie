#!/bin/bash
# Evaluate EN encoder-CRF grid (matched + clean protocols)
set -e
ROOT=/data/dual_noise
SEED=${1:-42}
LEVELS=(0 1 2 3 4 5)
QUALITY=(0.0 0.2 0.4 0.6 0.8 1.0)

cd $ROOT
source /data/anaconda3/etc/profile.d/conda.sh
conda activate dualnoise

RESULTS=$ROOT/experiments/results_en.csv
[ -f "$RESULTS" ] && rm -f "$RESULTS"
CLEAN_LS=$ROOT/data/processed/mtsamples_ls/L0_q1.0.json
export CUDA_VISIBLE_DEVICES=0

n=0
for lt in "${LEVELS[@]}"; do
  for q in "${QUALITY[@]}"; do
    cfg=$ROOT/experiments/logs/cfg_en_L${lt}_q${q}_seed${SEED}.json
    best=$ROOT/runs/mtsamples_L${lt}_q${q}_seed${SEED}/best/model.safetensors
    [ ! -f "$cfg" ] && continue
    [ ! -f "$best" ] && continue
    matched_ls=$ROOT/data/processed/mtsamples_ls/L${lt}_q${q}.json
    pred_m=$ROOT/experiments/predictions/encoder_crf_en_L${lt}_q${q}_le${lt}_seed${SEED}.json
    pred_c=$ROOT/experiments/predictions/encoder_crf_en_L${lt}_q${q}_le0_seed${SEED}.json
    if [ ! -f "$pred_m" ]; then
      python code/eval_cell.py --config "$cfg" --eval-ls "$matched_ls" --out "$pred_m" --result-row "$RESULTS" --baseline encoder_crf --lang en --lt $lt --rar $q --le $lt --seed $SEED 2>&1 | tail -1
      n=$((n+1))
    fi
    if [ ! -f "$pred_c" ]; then
      python code/eval_cell.py --config "$cfg" --eval-ls "$CLEAN_LS" --out "$pred_c" --result-row "$RESULTS" --baseline encoder_crf --lang en --lt $lt --rar $q --le 0 --seed $SEED 2>&1 | tail -1
      n=$((n+1))
    fi
  done
done

echo "[$(date +%H:%M:%S)] eval rounds: $n"
wc -l $RESULTS
