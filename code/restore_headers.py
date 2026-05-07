"""
DualNoise-ClinIE — CCKS 2019 Field-Header Restoration
======================================================
Converts CCKS 2019 Task 1 raw annotations into header-anchored KV format
expected by the rest of the pipeline.

Input format (CCKS 2019 official):
  {"originalText": "...", "entities": [{"label_type": "..", "start_pos": .., "end_pos": ..}, ...]}

Output format (DualNoise-ClinIE clean):
  {"id": "...", "text": "...", "labels": [{"key": "诊断", "value": "...", "span": [s, e],
                                            "entity_label": "..."}, ...]}

Usage
-----
python restore_headers.py \
    --input  data/raw/ccks2019/subtask1/train.json \
    --output data/processed/ccks_clean.jsonl
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

# Canonical Chinese inpatient field headers, ordered by usual appearance
HEADERS_ZH = [
    ("主诉",        re.compile(r"主\s*诉\s*[:：]?")),
    ("现病史",      re.compile(r"现\s*病\s*史\s*[:：]?")),
    ("既往史",      re.compile(r"既\s*往\s*史\s*[:：]?")),
    ("个人史",      re.compile(r"个\s*人\s*史\s*[:：]?")),
    ("家族史",      re.compile(r"家\s*族\s*史\s*[:：]?")),
    ("查体",        re.compile(r"(查\s*体|体\s*格\s*检\s*查|入\s*院\s*查\s*体)\s*[:：]?")),
    ("辅助检查",    re.compile(r"(辅\s*助\s*检\s*查|实\s*验\s*室\s*检\s*查|影\s*像\s*学\s*检\s*查)\s*[:：]?")),
    ("诊断",        re.compile(r"(诊\s*断|入\s*院\s*诊\s*断|出\s*院\s*诊\s*断)\s*[:：]?")),
    ("治疗经过",    re.compile(r"(治\s*疗\s*经\s*过|诊\s*疗\s*经\s*过|入\s*院\s*后\s*情\s*况)\s*[:：]?")),
]

CANONICAL_KEYS = [k for k, _ in HEADERS_ZH]

def find_section_boundaries(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, canonical_key) for each detected section."""
    spans = []
    for key, pat in HEADERS_ZH:
        for m in pat.finditer(text):
            spans.append((m.end(), key))  # content begins after header
    if not spans:
        return [(0, len(text), "未分段")]
    spans.sort()
    sections = []
    for i, (start, key) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        # back off to before next header
        sections.append((start, end, key))
    return sections

def assign_entity_to_section(entity_start: int, entity_end: int,
                             sections: List[Tuple[int, int, str]]) -> str:
    """Return canonical key of the section containing the entity, or 'AMBIGUOUS' if span crosses sections."""
    matched = []
    for start, end, key in sections:
        if entity_start >= start and entity_end <= end:
            matched.append(key)
    if len(matched) == 1: return matched[0]
    if len(matched) == 0:
        # entity may span before first detected header; assign to nearest preceding section
        for start, end, key in reversed(sections):
            if entity_start >= start: return key
        return "未分段"
    return "AMBIGUOUS"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="Drop entities whose span crosses section boundaries (recommended).")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_docs, n_dropped_docs = 0, 0
    n_entities, n_ambiguous, n_unsegmented = 0, 0, 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc.get("originalText") or doc.get("text") or ""
            entities = doc.get("entities") or doc.get("labels") or []
            if not text or not entities:
                continue
            n_docs += 1
            sections = find_section_boundaries(text)
            labels = []
            for ent in entities:
                # CCKS 2019 keys: start_pos, end_pos, label_type
                s = int(ent.get("start_pos") or ent.get("start") or 0)
                e = int(ent.get("end_pos") or ent.get("end") or s)
                lab = ent.get("label_type") or ent.get("type") or "UNK"
                key = assign_entity_to_section(s, e, sections)
                n_entities += 1
                if key == "AMBIGUOUS":
                    n_ambiguous += 1
                    if args.drop_ambiguous: continue
                if key == "未分段":
                    n_unsegmented += 1
                labels.append({
                    "key": key, "value": text[s:e],
                    "span": [s, e], "entity_label": lab,
                })
            if not labels:
                n_dropped_docs += 1
                continue
            fout.write(json.dumps({
                "id": doc.get("id") or f"ccks_{n_docs:05d}",
                "text": text,
                "labels": labels,
                "label_provenance": "ccks2019_gold",
            }, ensure_ascii=False) + "\n")

    print(f"docs: {n_docs} kept, {n_dropped_docs} dropped (no labels after restoration)")
    print(f"entities: {n_entities} total, {n_ambiguous} ambiguous ({n_ambiguous/max(n_entities,1):.1%}), "
          f"{n_unsegmented} unsegmented ({n_unsegmented/max(n_entities,1):.1%})")
    if n_ambiguous / max(n_entities, 1) > 0.15:
        print("WARNING: ambiguous-entity rate exceeds 15%. Section-header regex may need tuning.")

if __name__ == "__main__":
    main()
