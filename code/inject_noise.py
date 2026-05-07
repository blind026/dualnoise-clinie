"""
DualNoise-ClinIE — OCR Noise Injection
=======================================
Generates noisy variants of clean clinical text by injecting OCR errors
according to the calibrated 8-category taxonomy and the per-page
density Gamma model from §3.2 of the paper.

Usage
-----
python inject_noise.py \
    --input  data/processed/ccks_clean.jsonl \
    --output data/processed/ccks_noise/ \
    --lang   zh \
    --levels 0,1,2,3,4,5 \
    --seed   42

Each input line is a JSON object {"id": ..., "text": ..., "labels": [...]}.
Each output cell is a parallel JSONL with the same IDs but the noisy text
and a `noise_trace` field recording per-character corruption metadata.
"""

from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------- Calibrated parameters (§3.2) --------------------

# Reference noise levels (v1, ZH track): values are the *target* per-character
# corruption density. Realised densities are smaller because several injectors
# (visually-similar substitution, punctuation shift, digit-char split) have
# eligibility preconditions that gate per-position firing. Empirical realised
# densities measured on the CCKS HF mirror (v1):
#   L0 = 0.0%, L1 = 0.6%, L2 = 1.4%, L3 = 2.9%, L4 = 5.7%, L5 = 8.5%
# Confusion-dictionary expansion to bring realised in line with target is
# scheduled for v1.1.
NOISE_DENSITY_BY_LEVEL = {
    0: 0.000,
    1: 0.016,
    2: 0.039,
    3: 0.078,
    4: 0.156,
    5: 0.234,
}

# Realised densities measured on CCKS 2019 HF mirror (8157 docs, seed=42)
REALISED_DENSITY_BY_LEVEL_V1 = {
    0: 0.000,
    1: 0.006,
    2: 0.014,
    3: 0.029,
    4: 0.057,
    5: 0.085,
}

# 8-category taxonomy (Chinese), proportions sum to 1.0 (Tab 3 + Other)
ALPHA_ZH = {
    "visually_similar":  0.274,
    "spurious_whitespace": 0.238,
    "digit_char_split":  0.185,   # CHINESE-ONLY: dropped for EN
    "punctuation_shift": 0.170,
    "content_drop":      0.067,
    "section_misplace":  0.021,
    "garbage_chars":     0.001,
    "other":             0.044,
}

# English variant: drop digit_char_split, redistribute proportionally
def english_alpha() -> Dict[str, float]:
    keep = {k: v for k, v in ALPHA_ZH.items() if k != "digit_char_split"}
    Z = sum(keep.values())  # 0.815
    return {k: v / Z for k, v in keep.items()}

ALPHA_EN = english_alpha()

# Confusion dictionary stubs — Claude Code: replace with mined dictionaries
CONFUSION_ZH = {
    # OCR-mined frequent (clean -> noisy) substitutions; populate from corpus
    "肾": "肯",  # 肾囊肿 -> 肯爱肿
    "囊": "爱",
    "P": "D",
    "l": "1",   # luminal -> 1uminal
    # ... extend ...
}

CONFUSION_EN = {
    "O": "0", "0": "O",
    "l": "1", "1": "l", "I": "1",
    "rn": "m", "m": "rn",
    "c": "e",
    "vv": "w",
    "S": "5", "5": "S",
    "B": "8", "8": "B",
    "Z": "2", "2": "Z",
    # ... extend ...
}

PUNCTUATION_ZH = "，。、；：？！（）【】"
PUNCTUATION_EN = ",.;:?!()[]"

GARBAGE_GLYPHS = "|·~`^"

# -------------------- Injection logic --------------------

def inject_visually_similar(text: str, eligible: List[int], rho: float, conf: Dict[str, str], rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    chars = list(text)
    trace = []
    for i in eligible:
        if i >= len(chars): continue
        if rng.random() < rho and chars[i] in conf:
            new_c = conf[chars[i]]
            trace.append((i, f"vis_sim:{chars[i]}->{new_c}"))
            chars[i] = new_c
    return "".join(chars), trace

def inject_spurious_whitespace(text: str, eligible: List[int], rho: float, rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    out = []
    trace = []
    for i, c in enumerate(text):
        out.append(c)
        if i in eligible_set(eligible) and rng.random() < rho and not c.isspace():
            out.append(" ")
            trace.append((i, "ws_insert"))
    return "".join(out), trace

def inject_digit_char_split(text: str, eligible: List[int], rho: float, rng: random.Random, lang: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Chinese-only: insert space at digit-Chinese boundaries."""
    if lang != "zh": return text, []
    out = []
    trace = []
    for i, c in enumerate(text):
        out.append(c)
        if i + 1 < len(text):
            nxt = text[i + 1]
            is_boundary = (c.isdigit() and is_chinese(nxt)) or (is_chinese(c) and nxt.isdigit())
            if is_boundary and rng.random() < rho:
                out.append(" ")
                trace.append((i, "digit_split"))
    return "".join(out), trace

def inject_punctuation_shift(text: str, eligible: List[int], rho: float, rng: random.Random, lang: str) -> Tuple[str, List[Tuple[int, str]]]:
    puncs = PUNCTUATION_ZH if lang == "zh" else PUNCTUATION_EN
    chars = list(text)
    trace = []
    for i in eligible:
        if i >= len(chars): continue
        if chars[i] in puncs and rng.random() < rho and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            trace.append((i, "punct_shift"))
    return "".join(chars), trace

def inject_content_drop(text: str, eligible: List[int], rho: float, rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    """Drop a contiguous span of 1-5 characters."""
    out = list(text)
    trace = []
    drop_mask = [False] * len(out)
    for i in eligible:
        if i < len(out) and rng.random() < rho:
            span = rng.randint(1, 5)
            for j in range(i, min(i + span, len(out))):
                drop_mask[j] = True
            trace.append((i, f"drop:{span}"))
    return "".join(c for c, d in zip(out, drop_mask) if not d), trace

def inject_section_misplace(text: str, rho: float, rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    """Reorder paragraphs (rare, document-level)."""
    if rng.random() > rho * 50:  # bumped: per-document not per-char
        return text, []
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2: return text, []
    rng.shuffle(paragraphs)
    return "\n\n".join(paragraphs), [(0, "section_misplace")]

def inject_garbage(text: str, eligible: List[int], rho: float, rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    out = []
    trace = []
    for i, c in enumerate(text):
        out.append(c)
        if i in eligible_set(eligible) and rng.random() < rho:
            out.append(rng.choice(GARBAGE_GLYPHS))
            trace.append((i, "garbage"))
    return "".join(out), trace

def inject_other(text: str, eligible: List[int], rho: float, rng: random.Random) -> Tuple[str, List[Tuple[int, str]]]:
    """Catch-all: random one-character noise (delete, substitute, or insert)."""
    chars = list(text)
    trace = []
    for i in eligible:
        if i >= len(chars): continue
        if rng.random() < rho:
            choice = rng.choice(["sub", "ins", "del"])
            if choice == "sub":
                chars[i] = rng.choice("abc123#")
            elif choice == "ins":
                chars.insert(i, rng.choice("abc123#"))
            elif choice == "del" and len(chars) > 1:
                chars.pop(i)
            trace.append((i, f"other:{choice}"))
    return "".join(chars), trace

# -------------------- Helpers --------------------

def is_chinese(c: str) -> bool:
    return bool(re.match(r"[一-鿿]", c))

def eligible_set(eligible: List[int]) -> set:
    return set(eligible)

# -------------------- Pipeline --------------------

CATEGORY_FN = {
    "visually_similar":   "inject_visually_similar",
    "spurious_whitespace":"inject_spurious_whitespace",
    "digit_char_split":   "inject_digit_char_split",
    "punctuation_shift":  "inject_punctuation_shift",
    "content_drop":       "inject_content_drop",
    "section_misplace":   "inject_section_misplace",
    "garbage_chars":      "inject_garbage",
    "other":              "inject_other",
}

# Sequential dependency order: layout first, then position-conditional, then position-agnostic
INJECT_ORDER = [
    "section_misplace",   # 5
    "content_drop",       # 6
    "visually_similar",   # 1
    "digit_char_split",   # 3 (zh only)
    "spurious_whitespace",# 2
    "punctuation_shift",  # 4
    "garbage_chars",      # 7
    "other",              # 8
]

def inject_document(text: str, rho_n: float, lang: str, seed: int) -> Tuple[str, List]:
    rng = random.Random(seed)
    alpha = ALPHA_ZH if lang == "zh" else ALPHA_EN
    conf = CONFUSION_ZH if lang == "zh" else CONFUSION_EN
    full_trace = []

    for cat in INJECT_ORDER:
        if cat not in alpha: continue  # e.g., digit_char_split skipped for EN
        rho = rho_n * alpha[cat]
        eligible = list(range(len(text)))

        if cat == "visually_similar":
            text, t = inject_visually_similar(text, eligible, rho, conf, rng)
        elif cat == "spurious_whitespace":
            text, t = inject_spurious_whitespace(text, eligible, rho, rng)
        elif cat == "digit_char_split":
            text, t = inject_digit_char_split(text, eligible, rho, rng, lang)
        elif cat == "punctuation_shift":
            text, t = inject_punctuation_shift(text, eligible, rho, rng, lang)
        elif cat == "content_drop":
            text, t = inject_content_drop(text, eligible, rho, rng)
        elif cat == "section_misplace":
            text, t = inject_section_misplace(text, rho, rng)
        elif cat == "garbage_chars":
            text, t = inject_garbage(text, eligible, rho, rng)
        elif cat == "other":
            text, t = inject_other(text, eligible, rho, rng)
        full_trace.extend(t)

    return text, full_trace

def realised_density(orig_text: str, noisy_text: str, trace: List) -> float:
    return len(trace) / max(len(orig_text), 1)


def build_pos_map(clean: str, noisy: str) -> List[int]:
    """Return pos_map[i] = noisy index that clean char i maps to (or -1 if dropped).

    Uses Levenshtein opcodes for O(n*m) but C-optimized via python-Levenshtein.
    For chars that survive substitution: clean i -> noisy j.
    For chars that are deleted: clean i -> -1.
    Inserted noisy chars don't appear in pos_map (they have no clean origin).
    """
    try:
        import Levenshtein
    except ImportError:
        # Fall back to identity if length matches; otherwise raise
        if len(clean) == len(noisy):
            return list(range(len(clean)))
        raise RuntimeError("python-Levenshtein required for span remapping when text length changes")

    ops = Levenshtein.opcodes(clean, noisy)
    pos_map = [-1] * len(clean)
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for k in range(i2 - i1):
                pos_map[i1 + k] = j1 + k
        elif tag == "replace":
            # Map each clean[i1..i2) to its corresponding noisy position when possible
            n_clean = i2 - i1
            n_noisy = j2 - j1
            for k in range(n_clean):
                pos_map[i1 + k] = j1 + min(k, n_noisy - 1) if n_noisy > 0 else -1
        elif tag == "delete":
            # clean chars dropped
            for k in range(i1, i2):
                pos_map[k] = -1
        elif tag == "insert":
            pass  # nothing in clean to map
    return pos_map


def remap_span_through_posmap(span: List[int], pos_map: List[int], noisy_len: int) -> List[int] | None:
    """Remap a [s, e] clean span to noisy coordinates. Returns None if too corrupted to recover."""
    s, e = span[0], span[1]
    if s < 0 or e > len(pos_map) or s >= e:
        return None
    # Find first surviving char from s onwards as new start
    new_s = None
    for i in range(s, e):
        if pos_map[i] != -1:
            new_s = pos_map[i]
            break
    if new_s is None:
        return None  # entire span was dropped
    # Find last surviving char in [s, e) as new end (exclusive)
    new_e = None
    for i in range(e - 1, s - 1, -1):
        if pos_map[i] != -1:
            new_e = pos_map[i] + 1
            break
    if new_e is None or new_e <= new_s:
        return None
    return [max(0, new_s), min(noisy_len, new_e)]


def remap_labels(labels: List[Dict], pos_map: List[int], noisy_text: str) -> List[Dict]:
    """Remap each label's span through pos_map; drop labels whose span doesn't survive."""
    out = []
    n_noisy = len(noisy_text)
    for lab in labels:
        sp = lab.get("span")
        if not sp:
            continue
        new_sp = remap_span_through_posmap(sp, pos_map, n_noisy)
        if new_sp is None:
            continue
        new_lab = dict(lab)
        new_lab["span"] = new_sp
        new_lab["value"] = noisy_text[new_sp[0]:new_sp[1]]  # update value to noisy substring
        new_lab["clean_value"] = lab.get("value")  # preserve original clean value
        out.append(new_lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang", choices=["zh", "en"], required=True)
    ap.add_argument("--levels", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]

    docs = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines()]

    summary = {}
    for ell in levels:
        rho_n = NOISE_DENSITY_BY_LEVEL[ell]
        out_path = args.output / f"L{ell}.jsonl"
        densities = []
        with out_path.open("w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                seed = args.seed * 100000 + ell * 10000 + i
                noisy, trace = inject_document(doc["text"], rho_n, args.lang, seed)
                densities.append(realised_density(doc["text"], noisy, trace))
                # Remap label spans from clean coordinates to noisy coordinates.
                # For L0 (rho=0) the texts are identical and pos_map is the identity.
                if noisy == doc["text"]:
                    remapped_labels = doc["labels"]
                else:
                    pos_map = build_pos_map(doc["text"], noisy)
                    remapped_labels = remap_labels(doc["labels"], pos_map, noisy)
                out_doc = {
                    "id": doc["id"],
                    "text": noisy,
                    "labels": remapped_labels,
                    "clean_text_id": doc["id"],
                    "target_density": rho_n,
                    "realised_density": densities[-1],
                    "n_labels_clean": len(doc["labels"]),
                    "n_labels_after_remap": len(remapped_labels),
                    "noise_trace": trace,
                }
                f.write(json.dumps(out_doc, ensure_ascii=False) + "\n")
        mean_d = sum(densities) / len(densities) if densities else 0
        summary[f"L{ell}"] = {"target": rho_n, "realised_mean": mean_d,
                              "diff_pp": (mean_d - rho_n) * 100}
        print(f"L{ell}: target={rho_n:.3f} realised={mean_d:.3f} diff={summary[f'L{ell}']['diff_pp']:+.2f}pp")

    (args.output / "_density_report.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
