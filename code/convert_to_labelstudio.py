"""
DualNoise-ClinIE — Convert noisy cells to Label Studio JSON
============================================================
The BERT repo's training pipeline (BERT/pre_struct/kv_ner/) consumes
Label Studio JSON exports with KEY/VALUE/HOSPITAL spans. Our cell JSONL
format has only VALUE-style spans tagged by section header name. This
converter:

  1. For each cell document, scans the (already noisy) text for canonical
     section headers (主诉, 现病史, ..., 诊断, 治疗经过) using the
     restore_headers.py regex set, and emits each header match as a KEY span.
  2. Emits each label's span (already remapped to noisy coordinates by
     inject_noise.py post-fix) as a VALUE span.
  3. Wraps everything in the Label Studio task envelope expected by
     pre_struct/kv_ner/data_utils.py: data.ocr_text + annotations[*].result[*]
     where each result item has type='labels' and value.labels=['键名'|'值'].

Usage
-----
python convert_to_labelstudio.py \
    --input  data/processed/ccks_cells/L3_q0.6.jsonl \
    --output data/processed/ccks_ls/L3_q0.6.json \
    --lang   zh

Output is a single JSON array (Label Studio export format).
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

# ZH section-header regex (mirror of restore_headers.HEADERS_ZH)
HEADERS_ZH: List[Tuple[str, "re.Pattern"]] = [
    ("主诉",     re.compile(r"主\s*诉\s*[:：]?")),
    ("现病史",   re.compile(r"现\s*病\s*史\s*[:：]?")),
    ("既往史",   re.compile(r"既\s*往\s*史\s*[:：]?")),
    ("个人史",   re.compile(r"个\s*人\s*史\s*[:：]?")),
    ("家族史",   re.compile(r"家\s*族\s*史\s*[:：]?")),
    ("查体",     re.compile(r"(查\s*体|体\s*格\s*检\s*查|入\s*院\s*查\s*体)\s*[:：]?")),
    ("辅助检查", re.compile(r"(辅\s*助\s*检\s*查|实\s*验\s*室\s*检\s*查|影\s*像\s*学\s*检\s*查)\s*[:：]?")),
    ("诊断",     re.compile(r"(诊\s*断|入\s*院\s*诊\s*断|出\s*院\s*诊\s*断)\s*[:：]?")),
    ("治疗经过", re.compile(r"(治\s*疗\s*经\s*过|诊\s*疗\s*经\s*过|入\s*院\s*后\s*情\s*况)\s*[:：]?")),
]

# EN section-header patterns (canonical MTSamples-style)
HEADERS_EN: List[Tuple[str, "re.Pattern"]] = [
    ("chief_complaint",          re.compile(r"chief\s+complaint\s*:?", re.I)),
    ("history_present_illness",  re.compile(r"history\s+of\s+present\s+illness\s*:?", re.I)),
    ("past_medical_history",     re.compile(r"past\s+medical\s+history\s*:?", re.I)),
    ("physical_exam",            re.compile(r"physical\s+exam(?:ination)?\s*:?", re.I)),
    ("labs_imaging",             re.compile(r"(?:laboratory|labs|imaging|diagnostic\s+studies)\s*:?", re.I)),
    ("diagnosis",                re.compile(r"(?:diagnosis|assessment|impression)\s*:?", re.I)),
    ("treatment_course",         re.compile(r"(?:treatment|plan|hospital\s+course)\s*:?", re.I)),
]


def find_key_spans(text: str, lang: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, header_label) for every header match."""
    headers = HEADERS_ZH if lang == "zh" else HEADERS_EN
    spans = []
    for name, pat in headers:
        for m in pat.finditer(text):
            # match span is the header literal (e.g., "诊断:")
            spans.append((m.start(), m.end(), name))
    return spans


def build_ls_task(doc: dict, lang: str, task_id: int) -> dict:
    """Build a single Label Studio task from a cell doc."""
    text = doc["text"]
    title = doc.get("title", "")
    # KEY annotations: detected headers
    key_spans = find_key_spans(text, lang)
    # VALUE annotations: from labels (already in noisy coordinates after inject_noise.py fix)
    value_spans: List[Tuple[int, int]] = []
    for lab in doc.get("labels", []):
        sp = lab.get("span")
        if not sp or len(sp) != 2:
            continue
        s, e = sp
        if 0 <= s < e <= len(text):
            value_spans.append((s, e))

    results = []
    rid = 0
    for s, e, _name in key_spans:
        rid += 1
        results.append({
            "id": f"k{rid}",
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {"start": s, "end": e, "text": text[s:e], "labels": ["键名" if lang == "zh" else "KEY"]},
        })
    for s, e in value_spans:
        rid += 1
        results.append({
            "id": f"v{rid}",
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {"start": s, "end": e, "text": text[s:e], "labels": ["值" if lang == "zh" else "VALUE"]},
        })

    return {
        "id": task_id,
        "data": {
            "ocr_text": text,
            "category": title,
            "doc_id": doc.get("id", str(task_id)),
        },
        "annotations": [{"result": results, "was_cancelled": False}],
    }


def build_ls_task_flat(doc: dict, task_id: int) -> dict:
    """Flat-NER mode: each entity is an LS span labeled by its entity_label.

    Used for CCKS 2019 (ZH) where source docs are sentence-fragments with no
    detectable section headers — the paper's section-header KV framing does not
    apply. Each entity_label (disease_diagnosis, anatomical_site, ...) becomes
    a distinct NER class.
    """
    text = doc["text"]
    results = []
    rid = 0
    for lab in doc.get("labels", []):
        sp = lab.get("span")
        if not sp or len(sp) != 2:
            continue
        s, e = sp
        if not (0 <= s < e <= len(text)):
            continue
        # Prefer the per-document section/key label; fall back to scispaCy entity_label.
        # For EN (silver-labelled MTSamples) the section header is the more useful class
        # because scispaCy en_core_sci_md returns the single class "ENTITY".
        ent_label = lab.get("key") or lab.get("entity_label") or "ENTITY"
        rid += 1
        results.append({
            "id": f"e{rid}",
            "type": "labels",
            "from_name": "label",
            "to_name": "text",
            "value": {"start": s, "end": e, "text": text[s:e], "labels": [ent_label]},
        })
    return {
        "id": task_id,
        "data": {
            "ocr_text": text,
            "category": "",
            "doc_id": doc.get("id", str(task_id)),
        },
        "annotations": [{"result": results, "was_cancelled": False}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="Cell JSONL (one doc per line, output of annotation_error_sim.py or inject_noise.py)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Label Studio JSON array file")
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--schema", choices=["kv", "flat"], default="kv",
                    help="kv: synthesize KEY (header) + VALUE annotations (default, for docs with section headers); "
                         "flat: each entity uses its entity_label directly (for CCKS 2019 sentence-fragments)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    n_kept = 0
    n_total = 0
    n_keys_total = 0
    n_values_total = 0
    for i, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        doc = json.loads(line)
        n_total += 1
        if args.schema == "flat":
            task = build_ls_task_flat(doc, task_id=i)
            results = task["annotations"][0]["result"]
            n_v = len(results)
            n_keys_total += 0; n_values_total += n_v
            if n_v == 0:
                continue
        else:
            task = build_ls_task(doc, args.lang, task_id=i)
            results = task["annotations"][0]["result"]
            n_k = sum(1 for r in results if "KEY" in r["value"]["labels"] or "键名" in r["value"]["labels"])
            n_v = sum(1 for r in results if "VALUE" in r["value"]["labels"] or "值" in r["value"]["labels"])
            n_keys_total += n_k; n_values_total += n_v
            if n_k == 0 and n_v == 0:
                continue
        tasks.append(task)
        n_kept += 1

    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"docs: {n_kept}/{n_total} kept (with at least one annotation)")
    print(f"avg KEY annotations/doc: {n_keys_total / max(n_kept, 1):.2f}")
    print(f"avg VALUE annotations/doc: {n_values_total / max(n_kept, 1):.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
