"""
DualNoise-ClinIE — Qwen3-8B zero-shot eval via vllm offline LLM class.
Runs ONE noise level on ONE GPU; designed to be launched 8x in parallel.

Usage
-----
CUDA_VISIBLE_DEVICES=0 python eval_qwen_vllm.py \
    --model /data/dual_noise/models/qwen3-8b \
    --input data/processed/ccks_cells/L0_q1.0.jsonl \
    --output experiments/predictions/qwen3_zh_le0.json \
    --lang zh --max-docs 200
"""

from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

# Reuse the same prompt schema as eval_llm_zh_flat.py
sys.path.insert(0, str(Path(__file__).parent))
from eval_llm_zh_flat import (
    PROMPT_SYSTEM_ZH, PROMPT_SYSTEM_EN, SCHEMA_ZH, SCHEMA_EN,
    ENTITY_TYPES_ZH, ENTITY_TYPES_EN, build_prompt, to_predictions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load tokenizer for chat template
    tok = AutoTokenizer.from_pretrained(args.model)
    schema = SCHEMA_EN if args.lang == "en" else SCHEMA_ZH

    # Read input docs (cap at max-docs)
    docs = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        docs.append(json.loads(line))
        if len(docs) >= args.max_docs:
            break

    # Build prompts via chat template (Qwen uses ChatML)
    prompts = []
    for d in docs:
        msgs = build_prompt(d["text"], args.lang)
        # Use tokenizer's apply_chat_template; disable thinking explicitly via
        # `enable_thinking=False` passed through chat_template_kwargs
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)

    print(f"[qwen-vllm] {args.lang} cell: {len(prompts)} prompts; loading model...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # skip CUDA graphs for faster startup; small model so OK
    )
    print(f"[qwen-vllm] model loaded in {time.time() - t0:.1f}s; generating...", flush=True)

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sp, use_tqdm=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_failed = 0
    with args.output.open("w", encoding="utf-8") as f:
        for doc, o in zip(docs, outputs):
            raw = o.outputs[0].text if o.outputs else ""
            preds = []
            try:
                # Strip markdown-fenced JSON if present
                stripped = raw.strip()
                if stripped.startswith("```"):
                    stripped = stripped.split("```", 2)[1]
                    if stripped.lower().startswith("json"):
                        stripped = stripped[4:].lstrip()
                parsed = json.loads(stripped)
                preds = to_predictions(doc["text"], parsed, args.lang)
            except Exception:
                n_failed += 1
            f.write(json.dumps({"id": doc["id"], "predictions": preds, "raw_response": raw}, ensure_ascii=False) + "\n")
    print(f"[qwen-vllm] Done. {len(outputs)} processed, {n_failed} JSON parse failures. -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
