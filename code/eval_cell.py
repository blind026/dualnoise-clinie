"""
DualNoise-ClinIE — Per-cell encoder-CRF evaluator (matched-noise protocol)

Loads a trained mBERT-BiLSTM-CRF checkpoint and evaluates it on a Label Studio
JSON file (the eval cell), computing entity-level exact-match F1 per entity
type and overall.

Usage
-----
python eval_cell.py \
    --config experiments/logs/cfg_L3_q0.6_seed42.json \
    --eval-ls data/processed/ccks_ls/L3_q0.6.json \
    --out experiments/predictions/encoder_crf_zh_L3_q0.6_le3_seed42.json \
    --result-row experiments/results.csv \
    --baseline encoder_crf --lang zh --lt 3 --rar 0.6 --le 3 --seed 42
"""

from __future__ import annotations
import argparse, json, sys, os, csv
from pathlib import Path
from typing import Dict, List, Tuple

# Add BERT repo to path
BERT_ROOT = Path("/data/dual_noise/BERT")
if str(BERT_ROOT) not in sys.path:
    sys.path.insert(0, str(BERT_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pre_struct.kv_ner import config_io
from pre_struct.kv_ner.data_utils import build_bio_label_list, load_labelstudio_export
from pre_struct.kv_ner.dataset import TokenClassificationDataset, collate_batch
from pre_struct.kv_ner.metrics import char_spans
from pre_struct.kv_ner.modeling import BertCrfTokenClassifier


def predict_entities(model, tokenizer, samples, label2id, id2label, device, max_len=512):
    ds = TokenClassificationDataset(samples, tokenizer, label2id,
                                    max_seq_length=max_len,
                                    enable_chunking=True,
                                    chunk_size=500, chunk_overlap=50)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_batch)
    model.eval()
    out_per_task: Dict[str, List[Dict]] = {}
    text_per_task: Dict[str, str] = {}
    for batch in loader:
        with torch.no_grad():
            input_ids = batch.input_ids.to(device)
            mask = batch.attention_mask.to(device)
            out = model.predict(input_ids=input_ids, attention_mask=mask)
            # forward returns (loss=None, predictions) when labels=None; predictions are list of label-id lists
            preds = out["predictions"] if isinstance(out, dict) else out[1] if isinstance(out, tuple) else out
        for j, task_id in enumerate(batch.task_ids):
            offsets = batch.offset_mapping[j].tolist()
            attn = batch.attention_mask[j].tolist()
            text = batch.texts[j]
            cs = char_spans(preds[j], attn, offsets, id2label)
            # Convert chunk-relative spans to doc-relative using chunk_spans
            chunk_start, chunk_end = batch.chunk_spans[j]
            base_id = batch.original_task_ids[j]
            text_per_task[base_id] = batch.texts[j] if chunk_start == 0 else text_per_task.get(base_id, "")
            out_per_task.setdefault(base_id, [])
            for etype, s, e in cs:
                if 0 <= s < e:
                    out_per_task[base_id].append({"type": etype, "start": s + chunk_start, "end": e + chunk_start})
    # Dedup entities per task
    for tid in out_per_task:
        seen = set(); uniq = []
        for ent in sorted(out_per_task[tid], key=lambda x: (x["start"], x["end"], x["type"])):
            k = (ent["type"], ent["start"], ent["end"])
            if k in seen: continue
            seen.add(k); uniq.append(ent)
        out_per_task[tid] = uniq
    return out_per_task, text_per_task


def gold_from_samples(samples, full_text_lookup) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for s in samples:
        ents = []
        for e in s.entities:
            ents.append({"type": e.label.upper(), "start": e.start, "end": e.end})
        out[s.task_id] = ents
    return out


def f1_em(pred_doc: List[Dict], gold_doc: List[Dict]) -> Tuple[float, float, float, int, int, int]:
    pred_set = {(d["type"], d["start"], d["end"]) for d in pred_doc}
    gold_set = {(d["type"], d["start"], d["end"]) for d in gold_doc}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-ls", required=True, type=Path,
                    help="Label Studio JSON file (for the eval cell)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--result-row", required=True, type=Path)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--lt", required=True)
    ap.add_argument("--rar", required=True)
    ap.add_argument("--le", required=True)
    ap.add_argument("--seed", required=True)
    args = ap.parse_args()

    cfg = config_io.load_config(args.config)
    label_map = config_io.label_map_from(cfg)
    labels = build_bio_label_list(label_map)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    model_dir = cfg["train"]["output_dir"] + "/best"
    # Tokenizer is saved in a "tokenizer" subdir by BertCrfTokenClassifier.save_pretrained
    tokenizer_dir = model_dir + "/tokenizer"
    if not Path(tokenizer_dir + "/tokenizer_config.json").exists():
        tokenizer_dir = cfg["tokenizer_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = BertCrfTokenClassifier.from_pretrained(model_dir)
    # Use model's actual label2id (may differ from cfg if training added/removed labels)
    label2id = model.label2id
    id2label = {int(v): k for k, v in label2id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = load_labelstudio_export(args.eval_ls, label_map)
    # Limit to first 200 docs as held-out eval (otherwise we'd evaluate on training data)
    samples = samples[:200]

    pred_per, _ = predict_entities(model, tokenizer, samples, label2id, id2label, device,
                                   max_len=int(cfg.get("max_seq_length", 512)))
    gold_per = gold_from_samples(samples, {})

    # Aggregate metrics
    tp_total = fp_total = fn_total = 0
    rows_pred = []
    for s in samples:
        tid = s.task_id
        p = pred_per.get(tid, [])
        g = gold_per.get(tid, [])
        _, _, _, tp, fp, fn = f1_em(p, g)
        tp_total += tp; fp_total += fp; fn_total += fn
        rows_pred.append({
            "id": tid,
            "predictions": [{"key": e["type"], "value": s.text[e["start"]:e["end"]],
                              "span": [e["start"], e["end"]]} for e in p],
        })
    p_micro = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    r_micro = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    em_f1 = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_pred:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    args.result_row.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.result_row.exists()
    with args.result_row.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["baseline", "lang", "lt", "rar", "le", "seed", "em_f1", "precision", "recall",
                        "tp", "fp", "fn", "n_docs"])
        w.writerow([args.baseline, args.lang, args.lt, args.rar, args.le, args.seed,
                    f"{em_f1:.4f}", f"{p_micro:.4f}", f"{r_micro:.4f}",
                    tp_total, fp_total, fn_total, len(samples)])
    print(f"[eval] {args.baseline} lang={args.lang} lt={args.lt} rar={args.rar} le={args.le} seed={args.seed}: "
          f"EM-F1={em_f1:.4f} P={p_micro:.4f} R={r_micro:.4f} (n={len(samples)})")


if __name__ == "__main__":
    main()
