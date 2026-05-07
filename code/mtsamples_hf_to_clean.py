"""
DualNoise-ClinIE — MTSamples (HuggingFace mirror) → silver_label.py input format
================================================================================
The harishnair04/mtsamples mirror provides ~5000 transcribed clinical reports
with fields: description, medical_specialty, sample_name, transcription, keywords.

This script extracts the transcription field and emits a JSONL for downstream
silver-labelling and noise injection.

Usage
-----
python mtsamples_hf_to_clean.py \
    --hf-id harishnair04/mtsamples \
    --output data/raw/mtsamples/all.jsonl
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="harishnair04/mtsamples")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--min-chars", type=int, default=200, help="skip very short docs")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0; n_out = 0
    with args.output.open("w", encoding="utf-8") as f:
        ds = load_dataset(args.hf_id, split="train")
        for i, ex in enumerate(ds):
            n_in += 1
            text = (ex.get("transcription") or "").strip()
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({
                "id": f"mtsamples_{i:05d}",
                "text": text,
                "specialty": (ex.get("medical_specialty") or "").strip(),
                "sample_name": (ex.get("sample_name") or "").strip(),
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"Input docs: {n_in}, kept (≥{args.min_chars} chars): {n_out} → {args.output}")


if __name__ == "__main__":
    main()
