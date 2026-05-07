"""
DualNoise-ClinIE — 3-way ANOVA on encoder-CRF results
======================================================
Decomposes EM-F1 variance across the three factors:
  - lt (training noise)
  - rar (annotation quality)
  - le (evaluation noise)
plus their pairwise interactions.

Outputs experiments/anova_table.tex (LaTeX) and prints a summary.

Notes
-----
With one seed per cell we cannot estimate the residual variance, so we treat
each cell as a single sample and report sums-of-squares percentages without
significance tests. With seed=3 (planned for v2) the standard F-tests apply.
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np


def load_encoder_crf(results_csv: Path, lang: str = "zh"):
    rows = []
    with results_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["baseline"] != "encoder_crf" or r["lang"] != lang:
                continue
            try:
                rows.append({
                    "lt": int(r["lt"]),
                    "rar": float(r["rar"]),
                    "le": int(r["le"]),
                    "em_f1": float(r["em_f1"]),
                })
            except ValueError:
                continue
    return rows


def variance_decomp(rows):
    """Compute SS_total and SS for each main effect / interaction.

    Treats each (lt, rar, le) triple as a single observation.
    Uses Type-III-ish marginalization via group means.
    """
    if not rows:
        return None
    Y = np.array([r["em_f1"] for r in rows])
    grand_mean = Y.mean()
    SS_total = np.sum((Y - grand_mean) ** 2)
    if SS_total == 0:
        return None

    def ss_main(factor: str):
        """Sum of squares for main effect of `factor`."""
        levels = sorted({r[factor] for r in rows})
        ss = 0.0
        for lv in levels:
            grp = [r["em_f1"] for r in rows if r[factor] == lv]
            if not grp: continue
            ss += len(grp) * (np.mean(grp) - grand_mean) ** 2
        return ss

    def ss_inter(f1: str, f2: str):
        """Sum of squares for interaction between two factors (after subtracting main effects)."""
        l1 = sorted({r[f1] for r in rows})
        l2 = sorted({r[f2] for r in rows})
        m1 = {l: np.mean([r["em_f1"] for r in rows if r[f1] == l]) for l in l1}
        m2 = {l: np.mean([r["em_f1"] for r in rows if r[f2] == l]) for l in l2}
        ss = 0.0
        for a in l1:
            for b in l2:
                grp = [r["em_f1"] for r in rows if r[f1] == a and r[f2] == b]
                if not grp: continue
                pred = grand_mean + (m1[a] - grand_mean) + (m2[b] - grand_mean)
                ss += len(grp) * (np.mean(grp) - pred) ** 2
        return ss

    return {
        "SS_lt":  ss_main("lt"),
        "SS_rar": ss_main("rar"),
        "SS_le":  ss_main("le"),
        "SS_lt_rar":  ss_inter("lt", "rar"),
        "SS_lt_le":   ss_inter("lt", "le"),
        "SS_rar_le":  ss_inter("rar", "le"),
        "SS_total":   SS_total,
        "n":          len(rows),
        "grand_mean": grand_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()

    rows = load_encoder_crf(args.results, args.lang)
    if not rows:
        print(f"[err] no encoder_crf rows for lang={args.lang}"); return
    res = variance_decomp(rows)
    if res is None:
        print("[err] zero variance"); return

    SST = res["SS_total"]
    components = [
        ("L_e (eval noise)",            res["SS_le"]),
        ("ρ_a^r (annotation quality)",  res["SS_rar"]),
        ("L_t (train noise)",           res["SS_lt"]),
        ("L_t × L_e",                   res["SS_lt_le"]),
        ("L_t × ρ_a^r",                 res["SS_lt_rar"]),
        ("ρ_a^r × L_e",                 res["SS_rar_le"]),
    ]
    explained = sum(ss for _, ss in components)
    residual = max(0.0, SST - explained)

    print(f"\nANOVA decomposition ({args.lang}, n={res['n']}, mean F1={res['grand_mean']:.3f})")
    print(f"{'Component':<28} {'SS':>10} {'% of SS_total':>15}")
    print("-" * 56)
    for name, ss in components:
        print(f"{name:<28} {ss:>10.4f} {100*ss/SST:>13.1f}%")
    print(f"{'Residual':<28} {residual:>10.4f} {100*residual/SST:>13.1f}%")
    print(f"{'Total':<28} {SST:>10.4f} {100.0:>13.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lrr}\n\\toprule\n")
            f.write("Component & SS & \\% of SS\\textsubscript{total} \\\\\n\\midrule\n")
            for name, ss in components:
                f.write(f"{name} & {ss:.4f} & {100*ss/SST:.1f}\\% \\\\\n")
            f.write(f"Residual & {residual:.4f} & {100*residual/SST:.1f}\\% \\\\\n")
            f.write(f"\\midrule\nTotal & {SST:.4f} & 100.0\\% \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print(f"\nLaTeX table -> {args.out}")


if __name__ == "__main__":
    main()
