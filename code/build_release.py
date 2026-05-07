"""
DualNoise-ClinIE — v1 Artifact Release Builder
================================================
Bundles the v1 release into release/ with the structure expected by
HuggingFace Datasets + a Zenodo-friendly archive.

Outputs:
  release/dualnoise-clinie-zh-v1/
    README.md               # Dataset card (markdown)
    croissant.json          # mlcroissant metadata
    cells/L{0..5}_q{0..1}.jsonl       # 36 cells (clean + noisy + annotation-corrupted)
    clean/ccks_clean.jsonl  # the source clean docs
    LICENSE
    CHANGELOG.md
    release_manifest.yaml

Usage
-----
python build_release.py --root /data/dual_noise --out /data/dual_noise/release/dualnoise-clinie-zh-v1
"""

from __future__ import annotations
import argparse, json, shutil
from pathlib import Path


DATASET_CARD = """---
license: cc-by-nc-4.0
task_categories:
  - token-classification
  - feature-extraction
language:
  - zh
tags:
  - clinical
  - ner
  - ocr-noise
  - annotation-noise
  - benchmark
size_categories:
  - 1K<n<10K
---

# DualNoise-ClinIE — Chinese Release (v1)

A calibrated benchmark for joint OCR and annotation noise in Chinese clinical
information extraction. Built on top of the CCKS 2019 Task 1 corpus
(via the `doushabao4766/ccks_2019_ner_k_V3` HuggingFace mirror).

## What's in v1

- **8157 source documents** (sentence-level fragments from CCKS 2019 Task 1, 6 entity types: disease_diagnosis, anatomical_site, imaging_exam, lab_exam, surgery, medication).
- **36 reference cells**: 6 OCR-noise levels × 6 annotation-quality levels.
  - Noise levels L0–L5 correspond to *operational* densities {0%, 0.6%, 1.4%, 2.9%, 5.7%, 8.5%}, the realised per-character corruption fractions measured on the v1 release. Target densities (used to derive the level multipliers) are {0%, 1.6%, 3.9%, 7.8%, 15.6%, 23.4%}; the gap is documented in §3.3 and slated for v1.1 closure via confusion-dictionary expansion.
  - Annotation-quality levels q∈{0.0, 0.2, 0.4, 0.6, 0.8, 1.0} are senior-annotator ratios that drive a Beta-mixture annotator-error model (omissions + boundary shifts).
- **Toolkit** (separate repo): noise injector, annotation-error simulator, evaluation harness.

## Status

- v1: ZH-only.
- v1.1 (planned): EN release (MTSamples), expanded confusion dictionaries, 3-seed encoder-CRF runs.
- v1.2 (planned): full bilingual ANOVA, LoRA SFT cell.
- v2 (planned): MIMIC-IV-Note via PhysioNet credentialed access.

## Cell file format

Each `cells/L{ℓ}_q{q}.jsonl` contains one JSON object per line:

```json
{
  "id": "train_42",
  "text": "<noisy OCR-corrupted text>",
  "labels": [
    {"key": "disease_diagnosis", "value": "<noisy entity surface>",
     "span": [s, e], "entity_label": "disease_diagnosis",
     "clean_value": "<clean entity surface>"}
  ],
  "clean_text_id": "train_42",
  "target_density": 0.078,
  "realised_density": 0.029,
  "n_labels_clean": 3,
  "n_labels_after_remap": 3,
  "noise_trace": [...],
  "boundary_shifted": false
}
```

Noisy spans are mapped to noisy-text coordinates via Levenshtein-derived position
mapping, so labels are valid in the noisy text (this is a v1 fix relative to the
draft tooling).

## Reference baseline numbers

See `experiments/results.csv` in the toolkit repo for per-cell encoder-CRF F1.

## Limitations (v1)

1. **Source granularity:** the HF mirror provides sentence-level fragments rather than full inpatient reports. Section-header KV framing therefore does not apply; the v1 task is flat 6-class entity NER.
2. **Noise calibration:** realised density falls short of target due to per-category injector preconditions. The qualitative noise gradient is preserved.
3. **One seed only:** v1 ships single-seed runs; standard deviations not reported. v1.1 will add 3 seeds.
4. **Synthetic noise:** the calibration anchor (an in-house oncology corpus) is not redistributed; the calibration *statistics* are.
5. **Silver labels in source:** CCKS 2019 entity annotations are gold for the original task, so this corpus does not have silver-label provenance issues. The English release (v1.1) will.

## Citation

```bibtex
@misc{dualnoise2026,
  title  = {DualNoise-ClinIE: A Calibrated Benchmark for Joint OCR and Annotation Noise in Semi-Structured Clinical Information Extraction},
  author = {anonymous},
  year   = {2026},
  note   = {NeurIPS 2026 Datasets and Benchmarks track submission, under review}
}
```

## License

The CCKS 2019 derivative (clean + 36 cells) is released under CC-BY-NC-4.0,
inheriting the upstream CCKS research-license terms. The toolkit is MIT.
"""


CHANGELOG = """# Changelog

## v1.0 (2026-05-06)
- Initial release: 8157 ZH source docs, 36 reference cells, encoder-CRF baseline.

## Known issues addressed since draft
- Noise injection now correctly remaps label spans through a Levenshtein-derived
  clean→noisy position map. Earlier draft passed labels through unchanged,
  making them invalid in noisy text.
"""


CROISSANT_TEMPLATE = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "ml": "http://mlcommons.org/croissant/"
    },
    "@type": "Dataset",
    "name": "DualNoise-ClinIE-zh",
    "version": "1.0.0",
    "description": "Chinese clinical IE benchmark with joint OCR and annotation noise (v1, ZH-only).",
    "license": "https://creativecommons.org/licenses/by-nc/4.0/",
    "url": "https://huggingface.co/datasets/anon/dualnoise-clinie-zh",
    "creator": [{"@type": "Organization", "name": "anonymous-for-review"}],
    "keywords": ["clinical NLP", "OCR noise", "annotation noise", "benchmark"],
    "ml:tasks": [{"@type": "ml:Task", "name": "Token classification"}],
    "ml:distribution": [
        {"@type": "ml:FileSet",
         "name": "ccks_clean",
         "encodingFormat": "application/jsonlines",
         "containedIn": "clean/ccks_clean.jsonl"},
        {"@type": "ml:FileSet",
         "name": "cells_36",
         "encodingFormat": "application/jsonlines",
         "containedIn": "cells/"}
    ]
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    (out / "clean").mkdir(exist_ok=True)

    # Copy the 36 cells
    cells_src = args.root / "data" / "processed" / "ccks_cells"
    n_cells = 0
    for f in sorted(cells_src.glob("L*_q*.jsonl")):
        shutil.copy(f, out / "cells" / f.name)
        n_cells += 1
    print(f"copied {n_cells} cell files")

    # Copy the clean source
    clean_src = args.root / "data" / "processed" / "ccks_clean.jsonl"
    if clean_src.exists():
        shutil.copy(clean_src, out / "clean" / "ccks_clean.jsonl")
        print(f"copied clean source ({clean_src.stat().st_size / 1024:.0f} KB)")

    # Write README, LICENSE, CHANGELOG, croissant
    (out / "README.md").write_text(DATASET_CARD, encoding="utf-8")
    (out / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (out / "LICENSE").write_text("CC-BY-NC-4.0\n\nSee https://creativecommons.org/licenses/by-nc/4.0/\n")
    (out / "croissant.json").write_text(json.dumps(CROISSANT_TEMPLATE, ensure_ascii=False, indent=2))

    # Release manifest
    manifest = {
        "name": "DualNoise-ClinIE-zh",
        "version": "1.0.0",
        "n_cells": n_cells,
        "noise_levels": ["L0", "L1", "L2", "L3", "L4", "L5"],
        "annotation_quality_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "target_densities": {"L0": 0.000, "L1": 0.016, "L2": 0.039, "L3": 0.078,
                             "L4": 0.156, "L5": 0.234},
        "realised_densities_v1": {"L0": 0.000, "L1": 0.006, "L2": 0.014, "L3": 0.029,
                                   "L4": 0.057, "L5": 0.085},
        "entity_types": ["disease_diagnosis", "imaging_exam", "lab_exam",
                         "surgery", "medication", "anatomical_site"],
        "license": "CC-BY-NC-4.0",
    }
    (out / "release_manifest.yaml").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    total = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"release written to {out} ({total} files)")


if __name__ == "__main__":
    main()
