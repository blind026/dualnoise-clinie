"""
DualNoise-ClinIE — Annotation Error Simulator (§3.2 / §3.4)
============================================================
Two-component Beta mixture over per-annotator error rates:
  rho_a(i) ~ pi * Beta(alpha_s, beta_s) + (1 - pi) * Beta(alpha_j, beta_j)
where pi (= rho_a^r) is the senior-annotator ratio.

Per-instance corruption probability is sampled from the relevant component
to preserve within-group variance (not collapsed to group mean).
Each corruption is partitioned into omission vs boundary shift per Tab 4.

Usage
-----
python annotation_error_sim.py \
    --input data/processed/ccks_noise/L3.jsonl \
    --output data/processed/ccks_cells/L3_q0.6.jsonl \
    --rho-a-r 0.6 \
    --seed 42
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List

# Calibrated illustrative parameters (replace with empirical fits in Block 6)
ALPHA_S, BETA_S = 5.0, 73.0    # senior:  mean = 5/78  ≈ 6.4%
ALPHA_J, BETA_J = 5.0, 9.2     # junior:  mean = 5/14.2 ≈ 35.1%

# Per-group split between omission and boundary shift (Tab 4):
# senior: 4.8% omission, 1.64% boundary -> ratio 75/25
# junior: 19.2% omission, 15.92% boundary -> ratio 55/45
OMIT_RATIO = {"senior": 0.75, "junior": 0.55}

# Boundary shift offset std (chars), Tab 4
SIGMA_BOUNDARY = {"senior": 1.33, "junior": 2.93}

def sample_beta(alpha: float, beta: float, rng: random.Random) -> float:
    # Use python's betavariate (Marsaglia)
    return rng.betavariate(alpha, beta)

def corrupt_label(label: Dict, group: str, rng: random.Random) -> List[Dict]:
    """Return list of corrupted labels (empty if omitted, [shifted] if boundary, [original] if untouched)."""
    p_corrupt_mean = ALPHA_S/(ALPHA_S+BETA_S) if group == "senior" else ALPHA_J/(ALPHA_J+BETA_J)
    p_corrupt = sample_beta(*((ALPHA_S, BETA_S) if group == "senior" else (ALPHA_J, BETA_J)), rng)

    if rng.random() >= p_corrupt:
        return [label]  # not corrupted

    if rng.random() < OMIT_RATIO[group]:
        return []  # omission

    # boundary shift
    sigma = SIGMA_BOUNDARY[group]
    delta_start = int(round(rng.gauss(0, sigma)))
    delta_end   = int(round(rng.gauss(0, sigma)))
    new_label = dict(label)
    if "span" in label and isinstance(label["span"], (list, tuple)) and len(label["span"]) == 2:
        s, e = label["span"]
        new_label["span"] = [max(0, s + delta_start), max(s + delta_start + 1, e + delta_end)]
        new_label["boundary_shifted"] = True
    return [new_label]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--rho-a-r", type=float, required=True, help="Senior-annotator ratio in [0,1]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    n_orig, n_after = 0, 0
    n_omit, n_shift = 0, 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)
            new_labels = []
            for lab in doc["labels"]:
                n_orig += 1
                group = "senior" if rng.random() < args.rho_a_r else "junior"
                corrupted = corrupt_label(lab, group, rng)
                if not corrupted:
                    n_omit += 1
                else:
                    if corrupted[0].get("boundary_shifted"):
                        n_shift += 1
                    new_labels.extend(corrupted)
                    n_after += 1
            doc["labels"] = new_labels
            doc["rho_a_r"] = args.rho_a_r
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"orig={n_orig} after={n_after} omit={n_omit} ({n_omit/n_orig:.1%}) shift={n_shift} ({n_shift/n_orig:.1%})")

if __name__ == "__main__":
    main()
