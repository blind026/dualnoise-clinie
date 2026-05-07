#!/bin/bash
# Score all LLM predictions (gpt4o, deepseek, qwen3) and append to results CSV.
set -e
ROOT=/Users/haiyang/海心/论文/ToBeOrNotToBe
mkdir -p /tmp/dual_noise_subsets

# Build per-language gold subsets (first 200 docs of L_e cell at q=1.0)
for lang in zh en; do
  for L in 0 1 2 3 4 5; do
    if [ "$lang" = "zh" ]; then
      src=$ROOT/data/cells_for_llm/L${L}_q1.0.jsonl
    else
      src=$ROOT/data/cells_for_llm_en/L${L}_q1.0.jsonl
    fi
    [ ! -f "$src" ] && { echo "[gold MISS] $src"; continue; }
    head -200 "$src" > /tmp/dual_noise_subsets/gold_${lang}_le${L}.jsonl
  done
done

# Score
RESULTS=$ROOT/experiments/llm_results.csv
[ -f "$RESULTS" ] && rm -f "$RESULTS"

for model in gpt4o deepseek qwen3; do
  case "$model" in
    gpt4o) baseline="gpt-4o" ;;
    deepseek) baseline="deepseek-v4-pro" ;;
    qwen3) baseline="qwen3-8b" ;;
  esac
  for lang in zh en; do
    for L in 0 1 2 3 4 5; do
      pred=$ROOT/experiments/predictions/${model}_${lang}_le${L}_seed0.json
      [ ! -f "$pred" ] && continue
      n=$(wc -l < "$pred")
      [ "$n" -lt 50 ] && { echo "[skip $model $lang L$L: only $n preds]"; continue; }
      gold=/tmp/dual_noise_subsets/gold_${lang}_le${L}.jsonl
      [ ! -f "$gold" ] && continue
      python3 $ROOT/code/score.py \
        --predictions "$pred" --gold "$gold" \
        --output "$RESULTS" \
        --baseline "$baseline" --lang "$lang" \
        --lt NA --rar NA --le $L --seed 0 2>&1 | grep -oE '"em_f1": [0-9.]+|"pm_f1": [0-9.]+' | tr '\n' '|' | sed "s|^|[${baseline} ${lang} L${L}] |"
      echo
    done
  done
done

echo "---"
wc -l "$RESULTS"
