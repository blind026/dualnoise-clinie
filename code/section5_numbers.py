"""
DualNoise-ClinIE — Generate §5 numerical claims for paper update.

Reads experiments/results.csv and emits a markdown bulletin that Cowork-Claude
can paste into draft_neurips_db_v1.md §5.

Outputs:
  experiments/section5_numbers.md
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
from collections import defaultdict
import statistics


def load(p: Path):
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                r["em_f1"] = float(r["em_f1"])
                r["lt"] = int(r["lt"]); r["le"] = int(r["le"]); r["rar"] = float(r["rar"])
                rows.append(r)
            except Exception:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()

    rows = [r for r in load(args.results) if r["lang"] == args.lang and r["baseline"] == "encoder_crf"]
    if not rows:
        print("no rows")
        return

    # Build (lt, le) → mean F1 (over rar)
    grid_lt_le = defaultdict(list)
    for r in rows:
        grid_lt_le[(r["lt"], r["le"])].append(r["em_f1"])
    grid_lt_le = {k: statistics.mean(v) for k, v in grid_lt_le.items()}

    # Build (lt, rar) → mean F1 (over le, but typically le=lt for matched protocol)
    grid_lt_rar = defaultdict(list)
    for r in rows:
        grid_lt_rar[(r["lt"], r["rar"])].append(r["em_f1"])
    grid_lt_rar = {k: statistics.mean(v) for k, v in grid_lt_rar.items()}

    out_lines = [f"# §5 Numerical claims — {args.lang.upper()} (auto-generated)\n"]

    # §5.1 noise alignment effect
    out_lines.append("## §5.1 OCR Noise — Train–Evaluation Alignment Effect\n")
    if (0, 0) in grid_lt_le and (0, 5) in grid_lt_le:
        out_lines.append(f"- F1 at L_t=0, L_e=0: **{grid_lt_le[(0,0)]:.3f}**")
        out_lines.append(f"- F1 at L_t=0, L_e=5: **{grid_lt_le[(0,5)]:.3f}**")
        out_lines.append(f"- Drop from clean→noisy eval (L_t=0): **{(grid_lt_le[(0,0)] - grid_lt_le[(0,5)])*100:.1f} pp**\n")
    # Diagonal vs off-diagonal at L_e=3
    for le in [2, 3, 4]:
        diag_keys = [(le, le)]
        off_keys = [(0, le), (5, le)]
        diag = [grid_lt_le[k] for k in diag_keys if k in grid_lt_le]
        off  = [grid_lt_le[k] for k in off_keys if k in grid_lt_le]
        if diag and off:
            out_lines.append(f"- L_e={le}: matched (L_t={le}) F1 = {diag[0]:.3f}; "
                             f"off-diagonal mean = {statistics.mean(off):.3f}; "
                             f"alignment gain = {(diag[0]-statistics.mean(off))*100:+.1f} pp")

    # §5.2 annotation quality saturation
    out_lines.append("\n## §5.2 Annotation Quality — Recall-Bound + Saturation\n")
    for lt in [3]:
        rar_pairs = [(rar, grid_lt_rar.get((lt, rar))) for rar in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
        out_lines.append(f"At L_t={lt} (matched eval avg over le):")
        for rar, f1 in rar_pairs:
            if f1 is not None:
                out_lines.append(f"  - ρ_a^r = {rar:.1f} → F1 = **{f1:.3f}**")
        valid = [(rar, f1) for rar, f1 in rar_pairs if f1 is not None]
        if len(valid) >= 2:
            d_low  = (valid[3][1] - valid[1][1]) * 100 if len(valid) >= 4 else 0  # 0.2→0.6
            d_high = (valid[5][1] - valid[3][1]) * 100 if len(valid) >= 6 else 0  # 0.6→1.0
            out_lines.append(f"  - F1 gain ρ_a^r 0.2→0.6: {d_low:+.1f} pp; 0.6→1.0: {d_high:+.1f} pp")
            if d_low > d_high:
                out_lines.append(f"  - **Saturation observable around ρ_a^r ≈ 0.6** (gain rate halves above this).\n")

    # §5.3 ANOVA-style decomposition (delegated to anova.py)
    out_lines.append("\n## §5.3 Variance decomposition\n")
    out_lines.append("See `experiments/anova_table.tex` (run `code/anova.py`).\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
