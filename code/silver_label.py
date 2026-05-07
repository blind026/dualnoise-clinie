"""
DualNoise-ClinIE — MTSamples Silver Label Generation
=====================================================
Uses scispaCy + a regex section-header parser to produce silver KV labels
for MTSamples documents. Section schema mirrors §C.3.2 of the paper.

Usage
-----
python silver_label.py \
    --input  data/raw/mtsamples/all.jsonl \
    --output data/processed/mtsamples_clean.jsonl \
    --model  en_core_sci_md
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

import spacy

# Section-header regex; ALL-CAPS only to avoid matching body-text words.
# MTSamples uses uppercase for headers like "SUBJECTIVE:" and lowercase in body.
SECTION_PATTERNS = [
    (re.compile(r"\bCHIEF COMPLAINT\s*:"), "chief_complaint"),
    (re.compile(r"\b(?:HISTORY OF PRESENT ILLNESS|HPI|SUBJECTIVE|HISTORY)\s*:"), "history_of_present_illness"),
    (re.compile(r"\b(?:PAST MEDICAL HISTORY|PMH|PAST SURGICAL HISTORY|PSH|MEDICAL HISTORY|SURGICAL HISTORY)\s*:"), "past_medical_history"),
    (re.compile(r"\b(?:FAMILY HISTORY|SOCIAL HISTORY)\s*:"), "social_family_history"),
    (re.compile(r"\b(?:ALLERGIES|MEDICATIONS|CURRENT MEDICATIONS|HOME MEDICATIONS)\s*:"), "medications_allergies"),
    (re.compile(r"\b(?:PHYSICAL EXAM(?:INATION)?|EXAMINATION|EXAM|OBJECTIVE|VITAL SIGNS|HEENT|REVIEW OF SYSTEMS|ROS)\s*:"), "physical_examination"),
    (re.compile(r"\b(?:DIAGNOSTIC STUDIES|LABORATORY DATA|LABORATORY RESULTS|LABS|IMAGING|RADIOLOGY|EKG|ECG|CT SCAN|MRI|DOPPLER|ULTRASOUND)\s*:"), "diagnostic_studies"),
    (re.compile(r"\b(?:ASSESSMENT|DIAGNOSIS|IMPRESSION|PREOPERATIVE DIAGNOSIS|POSTOPERATIVE DIAGNOSIS|FINAL DIAGNOSIS|DIFFERENTIAL DIAGNOSIS)\s*:"), "assessment_diagnosis"),
    (re.compile(r"\b(?:PLAN|TREATMENT|PROCEDURE|OPERATION|HOSPITAL COURSE|DISCHARGE PLAN|RECOMMENDATIONS?|ANESTHESIA)\s*:"), "plan_treatment"),
]

def find_sections(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, key) for each detected section."""
    matches = []
    for pat, key in SECTION_PATTERNS:
        for m in pat.finditer(text):
            matches.append((m.end(), key))  # content starts after header
    if not matches:
        return []
    matches.sort()
    sections = []
    for i, (start, key) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        # back off `end` to start of the next header line
        nxt_header = text.rfind("\n", start, end)
        if nxt_header > start: end = nxt_header
        sections.append((start, end, key))
    return sections

def label_document(text: str, nlp) -> List[Dict]:
    """Section-level KV: each (key, value=full section content) per detected section."""
    labels = []
    sections = find_sections(text)
    if not sections:
        sections = [(0, len(text), "plan_treatment")]
    for start, end, key in sections:
        # Trim leading/trailing whitespace within the section span
        section_text = text[start:end]
        stripped = section_text.strip()
        if len(stripped) < 5:
            continue
        # Adjust span to the actual non-whitespace content
        offset_left = len(section_text) - len(section_text.lstrip())
        offset_right = len(section_text) - len(section_text.rstrip())
        v_start = start + offset_left
        v_end = end - offset_right
        labels.append({
            "key": key,
            "value": stripped,
            "span": [v_start, v_end],
            "entity_label": key,
        })
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="JSONL with {id, text} per line, OR raw MTSamples CSV (will be normalised).")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--model", default="en_core_sci_md")
    args = ap.parse_args()

    print(f"Loading scispaCy model {args.model} ...")
    nlp = spacy.load(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n, n_with_labels = 0, 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = doc["text"]
            labels = label_document(text, nlp)
            n += 1
            if labels: n_with_labels += 1
            fout.write(json.dumps({
                "id": doc["id"],
                "text": text,
                "labels": labels,
                "label_provenance": "silver_scispacy",
            }, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"  ... {n} docs processed, {n_with_labels} with at least one label")

    print(f"Done. {n} docs, {n_with_labels} with labels ({n_with_labels/n:.1%}).")

if __name__ == "__main__":
    main()
