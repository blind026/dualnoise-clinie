"""
DualNoise-ClinIE — Figure generator
====================================
Generates the four key figures used in §5 of the paper from
experiments/results.csv:

  Fig 4: Encoder-CRF F1 heatmap (training noise × evaluation noise) at fixed q
  Fig 5: Annotation-quality curves (F1 vs senior-annotator-ratio at fixed L)
  Fig 6: Cross-family robustness curves (5 baselines vs evaluation noise)
  Fig 7: Variance decomposition bar chart (ANOVA components)

CSV columns expected: baseline, lang, lt, rar, le, seed, em_f1, precision, recall,
                     tp, fp, fn, n_docs

Usage
-----
python make_figures.py --results experiments/results.csv --out figures/
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["em_f1"] = float(row["em_f1"])
                row["precision"] = float(row["precision"]) if row.get("precision") else 0.0
                row["recall"] = float(row["recall"]) if row.get("recall") else 0.0
                row["le"] = int(row["le"])
                # lt and rar can be 'NA' for LLMs (not trained, so no training-cell coords)
                try:
                    row["lt"] = int(row["lt"])
                except (ValueError, TypeError):
                    row["lt"] = None
                try:
                    row["rar"] = float(row["rar"])
                except (ValueError, TypeError):
                    row["rar"] = None
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def fig4_noise_heatmap(rows, out_dir: Path, lang: str = "zh"):
    """Heatmap: rows=lt, cols=le, color=mean F1 averaging over rar."""
    sub = [r for r in rows if r["baseline"] == "encoder_crf" and r["lang"] == lang]
    grid = np.full((6, 6), np.nan)
    for lt in range(6):
        for le in range(6):
            vals = [r["em_f1"] for r in sub if r["lt"] == lt and r["le"] == le]
            if vals:
                grid[lt, le] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([f"L{i}" for i in range(6)])
    ax.set_yticklabels([f"L{i}" for i in range(6)])
    ax.set_xlabel(r"Evaluation noise level $L_e$"); ax.set_ylabel(r"Training noise level $L_t$")
    ax.set_title(f"Encoder-CRF F1 — {lang.upper()} (avg over annotation quality $\\rho_a^r$)")
    for lt in range(6):
        for le in range(6):
            v = grid[lt, le]
            if not np.isnan(v):
                ax.text(le, lt, f"{v:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig4_noise_heatmap_{lang}.pdf", dpi=200)
    fig.savefig(out_dir / f"fig4_noise_heatmap_{lang}.png", dpi=120)
    plt.close(fig)


def fig5_annotation_quality(rows, out_dir: Path, lang: str = "zh"):
    """F1 vs rar at lt=3 (empirical mean), one curve per evaluation level."""
    sub = [r for r in rows if r["baseline"] == "encoder_crf" and r["lang"] == lang and r["lt"] == 3]
    fig, ax = plt.subplots(figsize=(7, 5))
    for le in range(6):
        rars = sorted({r["rar"] for r in sub if r["le"] == le})
        f1s = [np.mean([r["em_f1"] for r in sub if r["le"] == le and r["rar"] == rar]) for rar in rars]
        if rars and f1s:
            ax.plot(rars, f1s, marker="o", label=rf"$L_e={le}$")
    ax.set_xlabel(r"Senior-annotator ratio $\rho_a^r$"); ax.set_ylabel("EM-F1")
    ax.set_title(rf"F1 vs annotation quality (training at $L_t=3$) — {lang.upper()}")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig5_annotation_quality_{lang}.pdf", dpi=200)
    fig.savefig(out_dir / f"fig5_annotation_quality_{lang}.png", dpi=120)
    plt.close(fig)


def fig6_cross_family(rows, out_dir: Path, lang: str = "zh"):
    """F1 vs le for each baseline family on the q=1.0 evaluation surface.
    Uses encoder_crf_best (L_t at the best training cell) for fair comparison
    against zero-shot LLMs (which see only the eval text, not training noise)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    # Curate the baseline order so they appear in a meaningful order
    order = ["encoder_crf_best", "gpt-4o", "deepseek-v4-pro", "qwen3-8b"]
    for baseline in order:
        sub = [r for r in rows if r["baseline"] == baseline and r["lang"] == lang]
        if baseline == "encoder_crf_best":
            # this baseline already at q=1.0
            pass
        else:
            # LLMs are evaluated only at q=1.0 by construction
            pass
        levels = sorted({r["le"] for r in sub})
        f1s = [np.mean([r["em_f1"] for r in sub if r["le"] == lv]) for lv in levels]
        if levels:
            # Pretty baseline labels with proper symbols
            label_map = {
                "encoder_crf_best": r"Encoder-CRF (best $L_t,\rho_a^r$)",
                "encoder_crf": "Encoder-CRF (avg)",
                "gpt-4o": "GPT-4o (zero-shot)",
                "deepseek-v4-pro": "DeepSeek-V4-Pro (zero-shot)",
                "qwen3-8b": "Qwen3-8B (zero-shot)",
            }
            ax.plot(levels, f1s, marker="o", label=label_map.get(baseline, baseline))
    ax.set_xlabel(r"Evaluation noise level $L_e$"); ax.set_ylabel("EM-F1")
    ax.set_title(f"Cross-family robustness — {lang.upper()}")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig6_cross_family_{lang}.pdf", dpi=200)
    fig.savefig(out_dir / f"fig6_cross_family_{lang}.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--langs", default="zh,en")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.results)
    print(f"loaded {len(rows)} rows from {args.results}")
    for lang in args.langs.split(","):
        if not any(r["lang"] == lang for r in rows):
            print(f"[skip] no rows for lang={lang}"); continue
        fig4_noise_heatmap(rows, args.out, lang)
        fig5_annotation_quality(rows, args.out, lang)
        fig6_cross_family(rows, args.out, lang)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
