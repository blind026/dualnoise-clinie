#!/bin/bash
# Launch Qwen3-8B zero-shot eval across 8 GPUs in parallel.
# 12 cells (6 levels × 2 langs) split across 8 GPUs (some GPUs get 2 cells, others 1).
set -e
ROOT=/data/dual_noise
MODEL=$ROOT/models/qwen3-8b

cd $ROOT

# Build the work list: (lang, le, input_cell)
work=(
  "zh 0 ccks_cells/L0_q1.0.jsonl"
  "zh 1 ccks_cells/L1_q1.0.jsonl"
  "zh 2 ccks_cells/L2_q1.0.jsonl"
  "zh 3 ccks_cells/L3_q1.0.jsonl"
  "zh 4 ccks_cells/L4_q1.0.jsonl"
  "zh 5 ccks_cells/L5_q1.0.jsonl"
  "en 0 mtsamples_cells/L0_q1.0.jsonl"
  "en 1 mtsamples_cells/L1_q1.0.jsonl"
  "en 2 mtsamples_cells/L2_q1.0.jsonl"
  "en 3 mtsamples_cells/L3_q1.0.jsonl"
  "en 4 mtsamples_cells/L4_q1.0.jsonl"
  "en 5 mtsamples_cells/L5_q1.0.jsonl"
)

# Round-robin assign cells to GPUs 0..7
i=0
for entry in "${work[@]}"; do
  set -- $entry
  lang=$1; le=$2; cell=$3
  gpu=$((i % 8))
  out=$ROOT/experiments/predictions/qwen3_${lang}_le${le}_seed0.json
  log=$ROOT/experiments/logs/qwen3_${lang}_le${le}.log
  sess=q3_${lang}_l${le}
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 200 ]; then
    echo "skip $sess (already done)"
    i=$((i+1))
    continue
  fi
  echo "dispatch $sess on GPU $gpu (input: $cell)"
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "source /data/anaconda3/etc/profile.d/conda.sh && conda activate vllm_env && cd $ROOT && FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$gpu python code/eval_qwen_vllm.py --model $MODEL --input data/processed/$cell --output $out --lang $lang --max-docs 200 2>&1 | tee $log"
  i=$((i+1))
  # Stagger startup to avoid model-load contention
  sleep 2
done

sleep 3
echo "active sessions:"
tmux list-sessions 2>&1 | grep '^q3_' | wc -l
