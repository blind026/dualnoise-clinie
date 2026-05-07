"""
DualNoise-ClinIE — CCKS 2019 (HuggingFace mirror) → clean cell JSONL format
====================================================================================
NOTE: The HF mirror (doushabao4766/ccks_2019_ner_k_V3) provides sentence-fragment
documents (mean ~57 chars), not the full inpatient reports of the original CCKS 2019
Task 1 dump. As a result section headers (主诉:, 现病史:, etc.) appear in <7% of docs,
so we BYPASS restore_headers.py and emit the simpler clean-cell JSONL directly:

  {id, text, labels: [{key, value, span, entity_label}], label_provenance}

Each entity becomes a label with `key = entity_label` (e.g., "disease_diagnosis"),
matching the pivoted "flat NER" framing for the ZH track.
====================================================================================
The doushabao4766/ccks_2019_ner_k_V3 mirror of CCKS 2019 Task 1 stores docs as
{tokens: [char,char,...], ner_tags: [int,int,...]} BIO sequences.

This script:
  1. Loads the parquet shards (train/validation/test).
  2. Reconstructs originalText by joining tokens.
  3. Walks ner_tags to recover entity spans (B-X then I-X+).
  4. Emits the {originalText, entities: [{start_pos,end_pos,label_type}]} format
     that restore_headers.py expects, one JSON per line.

NER tag convention inferred from CCKS 2019 Task 1's six entity types:
  0 = O
  1/2 = B/I-disease_diagnosis (疾病和诊断)
  3/4 = B/I-imaging_exam      (影像检查)
  5/6 = B/I-lab_exam          (实验室检验)
  7/8 = B/I-surgery           (手术)
  9/10 = B/I-anatomical_site  (解剖部位)
  11/12 = B/I-medication      (药物)

Usage
-----
python ccks_hf_to_clean.py \
    --hf-id doushabao4766/ccks_2019_ner_k_V3 \
    --output data/raw/ccks2019/all.jsonl \
    --splits train,validation,test
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List

ENTITY_TYPES = [
    "O",
    "disease_diagnosis", "disease_diagnosis",
    "imaging_exam", "imaging_exam",
    "lab_exam", "lab_exam",
    "surgery", "surgery",
    "medication", "medication",
    "anatomical_site", "anatomical_site",
]
B_OR_I = ["O", "B", "I", "B", "I", "B", "I", "B", "I", "B", "I", "B", "I"]


def extract_entities(tokens: List[str], tags: List[int]):
    text = "".join(tokens)
    entities = []
    n = len(tokens)
    i = 0
    while i < n:
        t = tags[i]
        if t > 0 and B_OR_I[t] == "B":
            etype = ENTITY_TYPES[t]
            start = i
            i += 1
            while i < n and tags[i] > 0 and B_OR_I[tags[i]] == "I" and ENTITY_TYPES[tags[i]] == etype:
                i += 1
            end = i  # exclusive
            entities.append({
                "start_pos": start,
                "end_pos": end,
                "label_type": etype,
            })
        else:
            i += 1
    return text, entities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="doushabao4766/ccks_2019_ner_k_V3")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--splits", default="train,validation,test")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as f:
        for split in args.splits.split(","):
            ds = load_dataset(args.hf_id, split=split)
            for i, ex in enumerate(ds):
                tokens = ex["tokens"]
                tags = ex["ner_tags"]
                if not tokens:
                    continue
                text, entities = extract_entities(tokens, tags)
                if not entities:
                    continue
                # Convert HF entities -> cell-JSONL labels with key=entity_label (flat NER)
                labels = []
                for ent in entities:
                    s, e = ent["start_pos"], ent["end_pos"]
                    labels.append({
                        "key": ent["label_type"],
                        "value": text[s:e],
                        "span": [s, e],
                        "entity_label": ent["label_type"],
                    })
                f.write(json.dumps({
                    "id": f"{split}_{ex.get('id', i)}",
                    "text": text,
                    "labels": labels,
                    "label_provenance": "ccks2019_hf_mirror",
                    "split": split,
                }, ensure_ascii=False) + "\n")
                n += 1
            print(f"[{split}] wrote {n} cumulative docs")
    print(f"Done. Total docs: {n} → {args.output}")


if __name__ == "__main__":
    main()
