"""
Print a quick text-mode summary of the encoder-CRF grid: F1 by lt × rar (matched).
Also prints clean-eval F1 by lt × rar.
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()

    rows = []
    with args.results.open() as f:
        for r in csv.DictReader(f):
            if r["baseline"] != "encoder_crf" or r["lang"] != args.lang:
                continue
            try:
                rows.append({"lt": int(r["lt"]), "rar": float(r["rar"]),
                             "le": int(r["le"]), "f1": float(r["em_f1"])})
            except ValueError:
                continue

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["lt"], r["rar"], r["le"])].append(r["f1"])
    rar_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lt_levels = [0, 1, 2, 3, 4, 5]

    print(f"\n=== {args.lang.upper()} encoder-CRF — MATCHED protocol (le == lt) ===")
    print(f"{'lt\\rar':>6}", *[f"{q:>6.1f}" for q in rar_levels])
    for lt in lt_levels:
        cells = []
        for q in rar_levels:
            f1s = by_cell.get((lt, q, lt), [])
            cells.append(f"{f1s[0]:>6.3f}" if f1s else "  -   ")
        print(f"L{lt:<5}", *cells)

    print(f"\n=== {args.lang.upper()} encoder-CRF — CLEAN protocol (eval on L0_q1.0) ===")
    print(f"{'lt\\rar':>6}", *[f"{q:>6.1f}" for q in rar_levels])
    for lt in lt_levels:
        cells = []
        for q in rar_levels:
            f1s = by_cell.get((lt, q, 0), [])
            cells.append(f"{f1s[0]:>6.3f}" if f1s else "  -   ")
        print(f"L{lt:<5}", *cells)

    # Extras: train-eval alignment by le axis at fixed rar=1.0
    print(f"\n=== Train-eval alignment (rar=1.0): F1 at (lt, le) ===")
    print(f"{'lt\\le':>6}", *[f"  L{l}" for l in lt_levels])
    for lt in lt_levels:
        cells = []
        for le in lt_levels:
            f1s = by_cell.get((lt, 1.0, le), [])
            cells.append(f"{f1s[0]:>4.3f}" if f1s else " -  ")
        print(f"L{lt:<5}", *cells)


if __name__ == "__main__":
    main()
