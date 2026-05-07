"""
DualNoise-ClinIE — mBERT-BiLSTM-CRF Trainer
============================================
Trains the encoder-CRF reference baseline (§4.2) on a single (lt, rho_a^r) cell.

Input cell format (one line per document):
  {"id": "...", "text": "...", "labels": [{"key": "诊断", "value": "...",
                                             "span": [s, e],
                                             "entity_label": "..."}], ...}

The model produces BIO tags over the section-header keys (主诉/现病史/.../诊断/...),
which at inference time are decoded back to {key, value, span} predictions.

Usage
-----
python train_encoder_crf.py \
    --train-cell data/processed/ccks_cells/L3_q0.6.jsonl \
    --val-cell   data/processed/ccks_cells/L3_q0.6_val.jsonl \
    --eval-cell  data/processed/ccks_noise/L3_eval.jsonl \
    --gold-clean data/processed/ccks_gold_eval.jsonl \
    --output-dir experiments/runs/encoder_crf_zh_L3_q0.6_seed42 \
    --pred-out   experiments/predictions/encoder_crf_zh_L3_q0.6_le3_seed42.json \
    --lang zh --seed 42 --epochs 7 --batch-size 8

Hardware: single H20 / H200 / A100 / RTX 4090.
"""

from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF

# -------------------- BIO label space --------------------
# Section-header keys for ZH come from restore_headers.py CANONICAL_KEYS.
# For EN we use the silver_label.py header inventory.
KEYS_ZH = ["主诉", "现病史", "既往史", "个人史", "家族史", "查体", "辅助检查", "诊断", "治疗经过"]
KEYS_EN = ["chief_complaint", "history_present_illness", "past_medical_history",
           "physical_exam", "labs_imaging", "diagnosis", "treatment_course",
           "medications", "allergies", "social_history", "family_history"]


def build_label_set(keys: List[str]) -> Tuple[List[str], Dict[str, int]]:
    labels = ["O"]
    for k in keys:
        labels.append(f"B-{k}")
        labels.append(f"I-{k}")
    return labels, {l: i for i, l in enumerate(labels)}


# -------------------- Data loading --------------------
def load_jsonl(p: Path) -> List[dict]:
    return [json.loads(line) for line in open(p, "r", encoding="utf-8")]


def doc_to_bio(doc: dict, label2id: Dict[str, int], lang: str) -> List[int]:
    text = doc["text"]
    n = len(text)
    bio = ["O"] * n
    for lab in doc.get("labels", []):
        key = lab["key"]
        s, e = lab["span"]
        if s < 0 or e > n or s >= e:
            continue
        if f"B-{key}" not in label2id:
            continue
        bio[s] = f"B-{key}"
        for i in range(s + 1, e):
            bio[i] = f"I-{key}"
    return [label2id[t] for t in bio]


class CellDataset(Dataset):
    def __init__(self, docs: List[dict], tokenizer, label2id: Dict[str, int],
                 lang: str, max_len: int = 512, stride: int = 32):
        self.examples = []
        for doc in docs:
            text = doc["text"]
            labels = doc_to_bio(doc, label2id, lang)
            # Character-level tokenization (CCKS uses character-level)
            chars = list(text)
            # Sliding window on chars with stride
            for start in range(0, len(chars), max_len - stride):
                end = min(start + max_len - 2, len(chars))  # leave room for [CLS], [SEP]
                seg_chars = chars[start:end]
                seg_labels = labels[start:end]
                if not seg_chars:
                    break
                enc = tokenizer(seg_chars, is_split_into_words=True,
                                truncation=True, max_length=max_len,
                                padding="max_length", return_tensors="pt")
                word_ids = enc.word_ids(0)
                aligned = []
                prev = None
                for wid in word_ids:
                    if wid is None:
                        aligned.append(-100)
                    elif wid != prev:
                        aligned.append(seg_labels[wid] if wid < len(seg_labels) else -100)
                    else:
                        aligned.append(-100)
                    prev = wid
                self.examples.append({
                    "input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "labels": torch.tensor(aligned, dtype=torch.long),
                    "doc_id": doc["id"],
                    "char_start": start,
                    "char_end": end,
                    "n_chars_total": len(chars),
                })
                if end >= len(chars):
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "meta": [{k: b[k] for k in ("doc_id", "char_start", "char_end", "n_chars_total")} for b in batch],
    }
    return out


# -------------------- Model --------------------
class BertBiLSTMCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, lstm_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        d = self.encoder.config.hidden_size
        self.lstm = nn.LSTM(d, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h, _ = self.lstm(h)
        h = self.dropout(h)
        emissions = self.classifier(h)
        # CRF mask must be bool and have first timestep True
        mask = attention_mask.bool()
        if labels is not None:
            # Replace -100 with O (0) so CRF doesn't crash; mask is governed by attention_mask
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(emissions, crf_labels, mask=mask, reduction="mean")
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)


# -------------------- Training loop --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    keys = KEYS_ZH if args.lang == "zh" else KEYS_EN
    labels, label2id = build_label_set(keys)
    id2label = {i: l for l, i in label2id.items()}

    print(f"[info] {len(labels)} BIO tags")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_docs = load_jsonl(Path(args.train_cell))
    val_docs = load_jsonl(Path(args.val_cell)) if args.val_cell else train_docs[: max(1, len(train_docs) // 10)]
    print(f"[info] train={len(train_docs)} val={len(val_docs)}")

    train_ds = CellDataset(train_docs, tokenizer, label2id, args.lang, args.max_len, args.stride)
    val_ds = CellDataset(val_docs, tokenizer, label2id, args.lang, args.max_len, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertBiLSTMCRF(args.model_name, num_labels=len(labels)).to(device)

    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optim = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * 0.1), total_steps)

    best_f1 = -1.0
    patience = args.patience
    bad = 0
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            loss = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()
            total += loss.item()
        avg = total / max(1, len(train_loader))

        # Quick val: token-level F1 over BIO tags
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                gold = batch["labels"]
                mask = batch["attention_mask"].bool()
                for p_seq, g_seq, m_seq in zip(preds, gold, mask):
                    for j, (p, g) in enumerate(zip(p_seq, g_seq.tolist())):
                        if g == -100 or not m_seq[j]:
                            continue
                        if g != 0 or p != 0:  # ignore O-O
                            if p == g and g != 0:
                                tp += 1
                            elif p != g and g != 0:
                                fn += 1
                            elif p != g and p != 0:
                                fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"[epoch {epoch}] train_loss={avg:.4f} val_token_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; bad = 0
            torch.save({"model": model.state_dict(), "label2id": label2id, "args": vars(args)}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[info] early stop at epoch {epoch}")
                break

    # ---------- Inference on eval cell ----------
    if args.eval_cell and args.pred_out:
        print(f"[info] loading best ckpt {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path)["model"])
        model.eval()
        eval_docs = load_jsonl(Path(args.eval_cell))
        eval_ds = CellDataset(eval_docs, tokenizer, label2id, args.lang, args.max_len, args.stride)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        # Aggregate per-doc char-level predictions
        per_doc_chars: Dict[str, List[Optional[int]]] = {}
        n_chars: Dict[str, int] = {}
        with torch.no_grad():
            for batch in eval_loader:
                preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                for p_seq, meta, am in zip(preds, batch["meta"], batch["attention_mask"]):
                    doc_id = meta["doc_id"]; cs = meta["char_start"]; nT = meta["n_chars_total"]
                    if doc_id not in per_doc_chars:
                        per_doc_chars[doc_id] = [None] * nT
                        n_chars[doc_id] = nT
                    # Walk word_ids on the fly: rebuild from input_ids by re-tokenizing?
                    # Easier: take first non-special, non-pad subtoken predictions; mapping uses our enumeration.
                    # We didn't preserve word_ids per example — recompute via attention_mask + sentence ordering.
                    # Simpler approach: for char i in seg, take prediction at the first subtoken aligned earlier.
                    # We know each char became >=1 subtoken; use attention_mask count - 2 (CLS, SEP) as the
                    # per-char position by mapping in order. This works because is_split_into_words=True
                    # produces tokens in input order.
                    # Reconstruct word_ids: positions where token != CLS/SEP/PAD; then collapse.
                    pass
        # Save predictions in unified format. We use the simpler but reliable strategy of
        # re-tokenizing each doc once and mapping input_ids back to chars via word_ids.
        pred_out: List[dict] = []
        with torch.no_grad():
            for doc in eval_docs:
                text = doc["text"]; n = len(text)
                chars = list(text)
                tags = ["O"] * n
                for start in range(0, len(chars), args.max_len - args.stride):
                    end = min(start + args.max_len - 2, len(chars))
                    seg = chars[start:end]
                    if not seg:
                        break
                    enc = tokenizer(seg, is_split_into_words=True, truncation=True,
                                    max_length=args.max_len, padding="max_length", return_tensors="pt")
                    word_ids = enc.word_ids(0)
                    p = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))[0]
                    prev = None
                    for tok_idx, wid in enumerate(word_ids):
                        if wid is None or wid == prev:
                            continue
                        if tok_idx >= len(p):
                            break
                        char_idx = start + wid
                        if char_idx < n and tags[char_idx] == "O":
                            tags[char_idx] = id2label[p[tok_idx]]
                        prev = wid
                    if end >= len(chars):
                        break
                # Tags -> KV spans
                kv = []
                i = 0
                while i < n:
                    t = tags[i]
                    if t.startswith("B-"):
                        key = t[2:]; s = i; i += 1
                        while i < n and tags[i] == f"I-{key}":
                            i += 1
                        e = i
                        kv.append({"key": key, "value": text[s:e], "span": [s, e]})
                    else:
                        i += 1
                pred_out.append({"id": doc["id"], "predictions": kv})
        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.pred_out, "w", encoding="utf-8") as f:
            for r in pred_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[info] wrote {len(pred_out)} predictions to {args.pred_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cell", required=True)
    ap.add_argument("--val-cell", default=None)
    ap.add_argument("--eval-cell", default=None, help="Cell to predict on (for matched-noise protocol)")
    ap.add_argument("--pred-out", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--model-name", default="bert-base-multilingual-cased")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
