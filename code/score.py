"""
DualNoise-ClinIE — Unified Evaluation Harness
==============================================
Computes EM-F1, partial-match F1, key-only F1, and per-noise-category robustness
from prediction files in the unified JSON format.

Unified prediction file format (one line per document):
{
  "id": "<doc_id>",
  "predictions": [
    {"key": "诊断", "value": "右侧乳腺恶性肿瘤", "span": [12, 21]},
    ...
  ]
}

Usage
-----
python score.py \
    --predictions experiments/predictions/encoder_crf_zh_L3_q0.6_seed42.json \
    --gold        data/processed/ccks_gold_eval.jsonl \
    --output      experiments/results.csv \
    --baseline    encoder_crf \
    --lang        zh \
    --lt 3 --rar 0.6 --le 3 --seed 42
"""

from __future__ import annotations
import argparse, json, csv, re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Normalisation
def normalize_zh(s: str) -> str:
    # full-width -> half-width
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = re.sub(r"[\s　]+", "", s)  # strip all whitespace
    s = re.sub(r"[，。、；：？！（）【】「」]", "", s)  # strip CJK punctuation
    return s.lower()

def normalize_en(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def normalize(s: str, lang: str) -> str:
    return normalize_zh(s) if lang == "zh" else normalize_en(s)

def kv_set(items: List[Dict], lang: str, partial: bool = False) -> Set[Tuple[str, str]]:
    out = set()
    for it in items:
        k = it.get("key", "")
        v = normalize(it.get("value", ""), lang)
        if partial:
            # split value into tokens for partial overlap; use Jaccard on shingles
            v = " ".join(sorted(re.findall(r"\w+", v)))
        out.add((k, v))
    return out

def f1(pred: Set, gold: Set) -> Tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f

def key_only_set(items: List[Dict]) -> Set[str]:
    return {it.get("key", "") for it in items}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--lt", default="NA")
    ap.add_argument("--rar", default="NA")
    ap.add_argument("--le", default="NA")
    ap.add_argument("--seed", default="NA")
    ap.add_argument("--json-failure-rate", type=float, default=0.0,
                    help="Override failure rate; otherwise inferred from prediction file")
    args = ap.parse_args()

    preds = {}
    n_failed = 0
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n_failed += 1
            continue
        preds[obj["id"]] = obj.get("predictions", [])
    gold = {json.loads(l)["id"]: json.loads(l)["labels"]
            for l in args.gold.read_text(encoding="utf-8").splitlines()}

    n = len(gold)
    json_failure_rate = args.json_failure_rate or (n_failed / max(n, 1))

    em_p, em_r, em_f = 0.0, 0.0, 0.0
    pm_p, pm_r, pm_f = 0.0, 0.0, 0.0
    ko_p, ko_r, ko_f = 0.0, 0.0, 0.0
    counted = 0

    for doc_id, gold_labels in gold.items():
        pred_labels = preds.get(doc_id, [])
        gp, gg = kv_set(pred_labels, args.lang), kv_set(gold_labels, args.lang)
        p, r, f = f1(gp, gg); em_p += p; em_r += r; em_f += f
        ppp, ppg = kv_set(pred_labels, args.lang, partial=True), kv_set(gold_labels, args.lang, partial=True)
        p, r, f = f1(ppp, ppg); pm_p += p; pm_r += r; pm_f += f
        kp, kg = key_only_set(pred_labels), key_only_set(gold_labels)
        kkp, kkg = {(k, "") for k in kp}, {(k, "") for k in kg}
        p, r, f = f1(kkp, kkg); ko_p += p; ko_r += r; ko_f += f
        counted += 1

    em_p, em_r, em_f = em_p/counted, em_r/counted, em_f/counted
    pm_p, pm_r, pm_f = pm_p/counted, pm_r/counted, pm_f/counted
    ko_p, ko_r, ko_f = ko_p/counted, ko_r/counted, ko_f/counted

    row = {
        "baseline": args.baseline, "lang": args.lang,
        "lt": args.lt, "rar": args.rar, "le": args.le, "seed": args.seed,
        "n_docs": counted, "json_failure_rate": json_failure_rate,
        "em_p": em_p, "em_r": em_r, "em_f1": em_f,
        "pm_p": pm_p, "pm_r": pm_r, "pm_f1": pm_f,
        "ko_p": ko_p, "ko_r": ko_r, "ko_f1": ko_f,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    with args.output.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
