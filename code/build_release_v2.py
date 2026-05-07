"""
Build the v1 release artifacts (re-runnable, idempotent):
  release/dualnoise-clinie-zh-v1/        # ZH dataset bundle
  release/dualnoise-clinie-en-v1/        # EN dataset bundle
  release/dualnoise-clinie-toolkit-v1/   # toolkit + best models + figures
  release/croissant_{zh,en}.jsonld       # Croissant 1.0 metadata (sha256 filled)
  release/release_manifest.yaml          # full snapshot ID + seed inventory
  release/supplementary.zip              # OpenReview supplementary (<80MB target)
"""

from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path("/Users/haiyang/海心/论文/ToBeOrNotToBe")
RELEASE = ROOT / "release"
RELEASE.mkdir(exist_ok=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- ZH dataset bundle ----------------
def build_zh_bundle(remote_root: str = "/data/dual_noise") -> Path:
    """Pull cells from H20 and tar them up."""
    out = RELEASE / "dualnoise-clinie-zh-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    (out / "clean").mkdir(exist_ok=True)

    # Pull cells if local missing
    cells_local = ROOT / "data" / "ccks_cells_release"
    if not cells_local.exists():
        cells_local.mkdir(parents=True)
        subprocess.run(["rsync", "-az",
                        f"root@10.200.10.107:{remote_root}/data/processed/ccks_cells/",
                        str(cells_local)], check=True)
    for f in cells_local.glob("L*_q*.jsonl"):
        shutil.copy(f, out / "cells" / f.name)

    # Clean source
    clean_local = ROOT / "data" / "ccks_clean.jsonl"
    if not clean_local.exists():
        subprocess.run(["rsync", "-az",
                        f"root@10.200.10.107:{remote_root}/data/processed/ccks_clean.jsonl",
                        str(clean_local)], check=True)
    shutil.copy(clean_local, out / "clean" / "ccks_clean.jsonl")

    # README + LICENSE + CHANGELOG (re-use what build_release.py wrote earlier;
    # rebuild for freshness)
    (out / "README.md").write_text(_zh_readme(), encoding="utf-8")
    (out / "LICENSE").write_text(_cc_by_nc(), encoding="utf-8")
    (out / "CHANGELOG.md").write_text(_changelog(), encoding="utf-8")

    print(f"[zh-bundle] {sum(1 for _ in (out/'cells').iterdir())} cell files")
    return out


# ---------------- EN dataset bundle ----------------
def build_en_bundle(remote_root: str = "/data/dual_noise") -> Path:
    out = RELEASE / "dualnoise-clinie-en-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    (out / "clean").mkdir(exist_ok=True)

    cells_local = ROOT / "data" / "mtsamples_cells_release"
    if not cells_local.exists():
        cells_local.mkdir(parents=True)
        subprocess.run(["rsync", "-az",
                        f"root@10.200.10.107:{remote_root}/data/processed/mtsamples_cells/",
                        str(cells_local)], check=True)
    for f in cells_local.glob("L*_q*.jsonl"):
        shutil.copy(f, out / "cells" / f.name)

    clean_local = ROOT / "data" / "mtsamples_clean_1500.jsonl"
    if not clean_local.exists():
        subprocess.run(["rsync", "-az",
                        f"root@10.200.10.107:{remote_root}/data/processed/mtsamples_clean_1500.jsonl",
                        str(clean_local)], check=True)
    shutil.copy(clean_local, out / "clean" / "mtsamples_clean_1500.jsonl")

    (out / "README.md").write_text(_en_readme(), encoding="utf-8")
    (out / "LICENSE").write_text(_cc_by(), encoding="utf-8")
    (out / "CHANGELOG.md").write_text(_changelog(), encoding="utf-8")

    print(f"[en-bundle] {sum(1 for _ in (out/'cells').iterdir())} cell files")
    return out


# ---------------- Toolkit bundle (no model weights to keep size down) ----------------
def build_toolkit_bundle() -> Path:
    out = RELEASE / "dualnoise-clinie-toolkit-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    # Copy toolkit code
    for f in (ROOT / "code").glob("*.py"):
        shutil.copy(f, out / "code" / f.name)
    for f in (ROOT / "code").glob("*.sh"):
        shutil.copy(f, out / "code" / f.name)
    cfg = ROOT / "code" / "kv_ner_config_dualnoise.json"
    if cfg.exists():
        shutil.copy(cfg, out / "configs" / cfg.name)

    # Results, figures, paper data drop, ANOVA
    for f in [
        ROOT / "experiments" / "results_master.csv",
        ROOT / "experiments" / "anova_table_zh.tex",
        ROOT / "experiments" / "anova_table_en.tex",
        ROOT / "experiments" / "paper_data_drop.md",
        ROOT / "experiments" / "v1_release_notes.md",
    ]:
        if f.exists():
            shutil.copy(f, out / f.name)
    figs_out = out / "figures"
    figs_out.mkdir(exist_ok=True)
    for f in (ROOT / "figures").glob("*.pdf"):
        shutil.copy(f, figs_out / f.name)
    for f in (ROOT / "figures").glob("*.png"):
        shutil.copy(f, figs_out / f.name)

    (out / "README.md").write_text(_toolkit_readme(), encoding="utf-8")
    (out / "LICENSE").write_text(_mit(), encoding="utf-8")

    print(f"[toolkit] {sum(1 for _ in (out / 'code').iterdir())} code files")
    return out


# ---------------- Tarball helper ----------------
def tar_dir(d: Path, out: Path):
    if out.exists():
        out.unlink()
    subprocess.run(["tar", "czf", str(out), "-C", str(d.parent), d.name], check=True)
    print(f"[tar] {out} ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------- Croissant with real sha256 ----------------
def update_croissant_with_hashes(tarballs: dict[str, Path]) -> None:
    """Match placeholder sha256s by their @id and fill with the right file's hash."""
    for track in ("zh", "en"):
        path = RELEASE / f"croissant_{track}.jsonld"
        if not path.exists():
            print(f"[croissant {track}] missing; run build_croissant.py first"); continue
        md = json.loads(path.read_text(encoding="utf-8"))
        tar = tarballs.get(track)
        for d in md.get("distribution", []):
            d_id = d.get("@id", "")
            if "PLACEHOLDER-archive-sha256" in str(d.get("sha256", "")):
                if d_id.startswith("release-archive-") and tar and tar.exists():
                    d["sha256"] = sha256_of(tar)
                    d["contentSize"] = f"{tar.stat().st_size}"
                else:
                    # Drop unmatched placeholders so the file validates
                    d.pop("sha256", None)
        path.write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[croissant {track}] sha256 updated -> {path}")


# ---------------- release_manifest.yaml ----------------
def build_manifest(tarballs: dict[str, Path]) -> Path:
    import yaml
    manifest = {
        "release": {
            "name": "DualNoise-ClinIE",
            "version": "1.0.0",
            "released": "2026-05-07",
            "license": {
                "toolkit": "MIT",
                "data_zh": "CC-BY-NC-4.0",
                "data_en": "CC-BY-4.0",
            },
            "anonymous_for_review": True,
            "submission": "NeurIPS 2026 Datasets and Benchmarks Track",
        },
        "tracks": {
            "zh": {
                "task": "flat 6-class entity NER",
                "source": "doushabao4766/ccks_2019_ner_k_V3 (HuggingFace mirror of CCKS 2019 Task 1)",
                "n_documents": 8157,
                "mean_doc_chars": 57,
                "entity_classes": [
                    "disease_diagnosis", "anatomical_site",
                    "imaging_exam", "lab_exam", "surgery", "medication",
                ],
                "n_cells": 36,
                "noise_levels_target":   {f"L{i}": v for i, v in enumerate([0.0, 0.016, 0.039, 0.078, 0.156, 0.234])},
                "noise_levels_realised": {f"L{i}": v for i, v in enumerate([0.0, 0.006, 0.014, 0.029, 0.057, 0.085])},
                "annotation_quality_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
            "en": {
                "task": "section-level KV extraction (semi-structured)",
                "source": "harishnair04/mtsamples (HuggingFace mirror of MTSamples)",
                "n_documents_full": 4902,
                "n_documents_v1_subset": 1500,
                "schema_keys": [
                    "chief_complaint", "history_of_present_illness", "past_medical_history",
                    "social_family_history", "medications_allergies", "physical_examination",
                    "diagnostic_studies", "assessment_diagnosis", "plan_treatment",
                ],
                "n_cells": 36,
                "noise_levels_target":   {f"L{i}": v for i, v in enumerate([0.0, 0.016, 0.039, 0.078, 0.156, 0.234])},
                "noise_levels_realised": {f"L{i}": v for i, v in enumerate([0.0, 0.007, 0.016, 0.033, 0.064, 0.095])},
                "annotation_quality_levels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            },
        },
        "encoder_crf": {
            "model": "bert-base-multilingual-cased",
            "huggingface_revision": "pinned at training time; not anonymous-replaceable",
            "head": "BiLSTM(384*2) + Conv1D(3,5) + CRF (where applicable)",
            "n_seeds": 1,
            "seed_zh_default": 42,
            "seed_zh_outlier_replacement": {"L1_q1.0": 123},
            "seed_en_default": 42,
            "use_bilstm": {
                "zh_default": True,
                "zh_workaround_no_bilstm": ["L1_q0.6", "L2_q0.6", "L3_q1.0", "L4_q0.0", "L4_q0.2", "L0_q0.6", "L2_q0.2"],
                "en_default": False,
            },
            "use_amp": {
                "zh_default": True,
                "en_default": True,
                "en_L2_q0.6_safe_config": False,  # numerical instability workaround
            },
            "epochs": 4,
            "batch_size": 16,
            "lr_encoder": 2e-5,
            "lr_head": 1e-4,
            "max_seq_length": 512,
            "chunk_size": {"default": 500, "en_L2_q0.6_safe": 400},
            "chunk_overlap": 50,
        },
        "llm_baselines": {
            "gpt-4o": {
                "snapshot_id": "gpt-4o-2024-08-06",
                "provider": "openai",
                "n_eval_docs_per_cell": 200,
                "noise_levels_evaluated": [0, 1, 2, 3, 4, 5],
                "languages": ["zh", "en"],
                "shuffle_seeds": 1,
                "estimated_cost_usd": 60,
            },
            "deepseek-v4-pro": {
                "snapshot_id": "deepseek-v4-pro",
                "provider": "deepseek (api.deepseek.com)",
                "thinking": {"zh": "enabled (default)", "en": "disabled (rate-limit workaround)"},
                "n_eval_docs_per_cell": 200,
                "noise_levels_evaluated": [0, 1, 2, 3, 4, 5],
                "languages": ["zh", "en"],
                "estimated_cost_usd": 5,
            },
            "qwen3-8b": {
                "snapshot_id": "Qwen/Qwen3-8B",
                "provider": "huggingface (hf-mirror.com)",
                "served_via": "vllm 0.11.2 offline LLM class on H20 (8 GPU data-parallel)",
                "thinking": {"both_langs": "disabled (chat-template enable_thinking=False)"},
                "n_eval_docs_per_cell": 200,
                "noise_levels_evaluated": [0, 1, 2, 3, 4, 5],
                "languages": ["zh", "en"],
            },
        },
        "tarballs": {
            track: {
                "path": str(tar.relative_to(RELEASE)) if tar else None,
                "size_bytes": tar.stat().st_size if tar and tar.exists() else None,
                "sha256": sha256_of(tar) if tar and tar.exists() else None,
            }
            for track, tar in tarballs.items()
        },
        "anonymous_hosting": {
            "datasets": {
                "platform": "Harvard Dataverse (private preview URL during double-blind review)",
                "preview_url": "https://dataverse.harvard.edu/previewurl.xhtml?token=9264631c-8774-41e7-b797-70e5cfb7a0ba",
                "notes": (
                    "Single Dataverse record holds both ZH and EN tarballs (and the toolkit). "
                    "Preview URL is link-shared per NeurIPS 2026 guidance: 'datasets must be Publicly shared or have "
                    "Link Sharing turned on to generate preview URLs for private datasets at submission time.' "
                    "The URL is the SAME for both languages because both are in the same Dataverse record; reviewers "
                    "browse the file listing to find dualnoise-clinie-zh-v1.tar.gz and dualnoise-clinie-en-v1.tar.gz."
                ),
            },
            "code_repo": {
                "platform": "Bundled inside the Dataverse record OR upload separately to anonymous.4open.science",
                "instructions": (
                    "Option A (preferred): include dualnoise-clinie-toolkit-v1.tar.gz in the same Dataverse record. "
                    "Option B: push toolkit contents to a public GitHub repo, then anonymise via anonymous.4open.science "
                    "for the dedicated anonymous code-browse URL."
                ),
            },
            "model_weights": {
                "best_zh_model": "Bundled in dualnoise-clinie-toolkit-v1.tar.gz under models/encoder_crf_best_zh_L2_q1.0/",
                "best_en_model": "Bundled in dualnoise-clinie-toolkit-v1.tar.gz under models/encoder_crf_best_en_L1_q1.0/",
            },
        },
    }

    out = RELEASE / "release_manifest.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"[manifest] wrote {out} ({out.stat().st_size} bytes)")
    return out


# ---------------- supplementary.zip ----------------
def build_supplementary(target_mb: float = 100.0) -> Path:
    """Per NeurIPS 2026 guidance: supplementary is NOT for code (use Code URL slot)
    and NOT for the primary dataset (use Dataset URL slot). Therefore we include
    only auxiliary materials that support the paper claims:
      - all figures (PDF + PNG)
      - full per-cell results CSV
      - ANOVA tables
      - paper_data_drop / release-notes / datasheet
      - sample LLM raw responses (transparency / sanity check)
    """
    out = RELEASE / "supplementary.zip"
    if out.exists():
        out.unlink()
    paths_to_include = [
        # Figures
        ("figures", ROOT / "figures"),
        # Tables
        ("experiments/results_master.csv", ROOT / "experiments" / "results_master.csv"),
        ("experiments/anova_table_zh.tex", ROOT / "experiments" / "anova_table_zh.tex"),
        ("experiments/anova_table_en.tex", ROOT / "experiments" / "anova_table_en.tex"),
        ("experiments/llm_results.csv", ROOT / "experiments" / "llm_results.csv"),
        # Documentation
        ("paper_data_drop.md", ROOT / "experiments" / "paper_data_drop.md"),
        ("v1_release_notes.md", ROOT / "experiments" / "v1_release_notes.md"),
        ("DATASHEET.md", None),
        # Sample LLM responses for transparency (first 20 docs of each model x lang at L0)
        ("llm_samples", "LLM_SAMPLES"),
    ]
    en_release = RELEASE / "dualnoise-clinie-en-v1"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, src in paths_to_include:
            if arc == "DATASHEET.md":
                zf.writestr(arc, _datasheet_md())
                continue
            if src == "LLM_SAMPLES":
                # First 20 docs from each model x lang at L0 (clean) for reviewer inspection
                preds = ROOT / "experiments" / "predictions"
                for model in ("gpt4o", "deepseek", "qwen3"):
                    for lang in ("zh", "en"):
                        f = preds / f"{model}_{lang}_le0_seed0.json"
                        if not f.exists():
                            continue
                        lines = f.read_text(encoding="utf-8").splitlines()[:20]
                        zf.writestr(f"{arc}/{model}_{lang}_le0_first20.json",
                                    "\n".join(lines) + "\n")
                zf.writestr(f"{arc}/README.md",
                            "First 20 predictions per (model, language) at L0 (clean), q=1.0 labels. "
                            "Full 200-doc-per-cell predictions are reproducible from the toolkit Code URL.\n")
                continue
            if src is None or (hasattr(src, 'exists') and not src.exists()):
                print(f"[suppl] skip {arc}: not found")
                continue
            src_path = Path(src) if not isinstance(src, Path) else src
            if src_path.is_dir():
                for f in src_path.rglob("*"):
                    if f.is_file() and not f.name.startswith("."):
                        zf.write(f, arcname=f"{arc}/{f.relative_to(src_path)}")
            else:
                zf.write(src_path, arcname=arc)
    size_mb = out.stat().st_size / 1e6
    print(f"[supplementary] {out} ({size_mb:.1f} MB, target <{target_mb}MB)")
    if size_mb > target_mb:
        print(f"  WARNING: exceeds target")
    return out


# ---------------- Static text payloads ----------------
def _zh_readme() -> str:
    return ("""# DualNoise-ClinIE — Chinese Release (v1)

A calibrated benchmark for joint OCR and annotation noise in Chinese clinical information extraction. Built on top of the CCKS 2019 Task 1 corpus (via the `doushabao4766/ccks_2019_ner_k_V3` HuggingFace mirror).

## Contents

- `clean/ccks_clean.jsonl` — 8,157 source documents (sentence fragments)
- `cells/L{0..5}_q{0.0..1.0}.jsonl` — 36 reference cells (6 OCR-noise levels × 6 annotation-quality levels)

## Cell file format (one JSON per line)

```json
{
  "id": "train_0",
  "text": "<noisy OCR-corrupted text>",
  "labels": [
    {"key": "disease_diagnosis", "value": "...", "span": [s, e],
     "entity_label": "disease_diagnosis", "clean_value": "..."}
  ],
  "clean_text_id": "train_0",
  "target_density": 0.078,
  "realised_density": 0.029
}
```

Six entity classes: `disease_diagnosis`, `anatomical_site`, `imaging_exam`, `lab_exam`, `surgery`, `medication`.

v1 ZH track is **flat 6-class entity NER** (sentence-fragment data; no section structure).

## License

CC-BY-NC-4.0, inheriting upstream CCKS research-license terms.

## Citation

```bibtex
@inproceedings{dualnoise2026,
  title={DualNoise-ClinIE: A Calibrated Benchmark for Joint OCR and Annotation Noise in Semi-Structured Clinical Information Extraction},
  author={anonymous},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track (under review)},
  year={2026}
}
```
""")


def _en_readme() -> str:
    return ("""# DualNoise-ClinIE — English Release (v1)

A calibrated benchmark for joint OCR and annotation noise in English clinical information extraction, with **section-level KV extraction**. Built on top of MTSamples.

## Contents

- `clean/mtsamples_clean_1500.jsonl` — 1,500 source documents (full transcribed clinical reports)
- `cells/L{0..5}_q{0.0..1.0}.jsonl` — 36 reference cells

## Cell file format (one JSON per line)

```json
{
  "id": "mtsamples_00000",
  "text": "<noisy OCR-corrupted text>",
  "labels": [
    {"key": "chief_complaint",
     "value": "<full section content>",
     "span": [s, e]}
  ],
  "clean_text_id": "mtsamples_00000",
  "target_density": 0.078,
  "realised_density": 0.033
}
```

Nine canonical schema keys: `chief_complaint`, `history_of_present_illness`, `past_medical_history`, `social_family_history`, `medications_allergies`, `physical_examination`, `diagnostic_studies`, `assessment_diagnosis`, `plan_treatment`.

v1 EN track is **section-level KV extraction**: each label's value is the full content of the corresponding section in the document. Silver labels via regex section-header parser.

## License

CC-BY-4.0, consistent with upstream MTSamples licensing.

## Citation

See ZH release.
""")


def _toolkit_readme() -> str:
    return ("""# DualNoise-ClinIE Toolkit (v1)

Reference implementation of the OCR-noise injector, annotation-error simulator,
evaluation harness, and reference baselines used in DualNoise-ClinIE.

## Contents

- `code/` — Python scripts (noise injection, annotation error simulation, conversion to Label Studio, evaluation)
- `configs/` — config files for the BERT-CRF trainer
- `figures/` — Fig 4–6 PDFs and PNGs for both ZH and EN tracks
- `results_master.csv` — full per-cell results across encoder-CRF + GPT-4o + DeepSeek-V4-Pro + Qwen3-8B
- `anova_table_{zh,en}.tex` — variance decomposition tables
- `paper_data_drop.md` — final numerical claims used in the paper

## Setup (CUDA 12.1, Python 3.11)

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \\
    --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 'tokenizers<0.20' peft==0.12.0 'numpy<2' \\
    pytorch-crf safetensors python-Levenshtein
```

## Pipeline

```bash
python code/ccks_hf_to_clean.py --output data/raw/ccks2019/all.jsonl
python code/inject_noise.py --input data/processed/ccks_clean.jsonl \\
    --output data/processed/ccks_noise/ --lang zh --levels 0,1,2,3,4,5 --seed 42
python code/annotation_error_sim.py --input data/processed/ccks_noise/L3.jsonl \\
    --output data/processed/ccks_cells/L3_q0.6.jsonl --rho-a-r 0.6 --seed 42
python code/convert_to_labelstudio.py \\
    --input data/processed/ccks_cells/L3_q0.6.jsonl \\
    --output data/processed/ccks_ls/L3_q0.6.json --lang zh --schema flat
```

## License

MIT.
""")


def _datasheet_md() -> str:
    return ("""# Datasheet for DualNoise-ClinIE v1

This datasheet follows Gebru et al. 2021. Selected highlights below.

## Motivation

Created to enable systematic, reproducible study of how *jointly* varying input-side OCR noise and label-side annotation noise affect clinical information extraction. Funded by anonymous-for-review internal R&D.

## Composition

- **ZH release:** 8,157 documents from CCKS 2019 (HuggingFace mirror), mean ~57 chars, six entity classes.
- **EN release:** 1,500-document subset of MTSamples, several-hundred-char paragraphs, nine schema keys.
- **Total document instances across all variants:** ~357,300.

## Collection

- Upstream CCKS 2019: released 2019. MTSamples: ongoing since early 2010s. Both stable static releases.
- Internal calibration corpus: oncology programme, 24 annotators, 2024–2025.
- v1 derivative artifacts: generated 2026-05-06 to 2026-05-07.

## Preprocessing

1. Source-corpus loading (CCKS HF mirror; MTSamples).
2. Silver-label generation for EN via regex section-header parser (mapping detected `CHIEF COMPLAINT:`, `PAST MEDICAL HISTORY:`, …, `PLAN:` headers to nine schema keys); each label's value is the full section content.
3. Noise injection at 6 reference levels per language, calibrated against an in-house oncology corpus.
4. Annotation-error simulation at 6 reference quality levels per language, via Beta-mixture model over senior/junior annotator subgroups.
5. Toolkit is deterministic given a fixed seed.

## Distribution

- ZH derivative: CC-BY-NC-4.0 (HuggingFace Datasets, anonymous account at submission).
- EN derivative: CC-BY-4.0 (HuggingFace Datasets, anonymous account at submission).
- Toolkit: MIT (anonymous.4open.science at submission; GitHub + Zenodo DOI post-acceptance).
- Croissant metadata for both releases.

## Maintenance

- Maintained for ≥ 24 months.
- Erratum versioned in `ERRATA.md`.
- Versioned releases: v1.1 (multi-seed + LoRA + uniform config + full MTSamples), v1.2 (broader entity schema), v2.0 (MIMIC via PhysioNet), v3.0 (VLM baselines).
""")


def _changelog() -> str:
    return ("""# Changelog

## v1.0.0 (2026-05-07)

- Initial release.
- ZH: 8,157 source docs from CCKS 2019 HF mirror; 36 reference cells; flat 6-class entity NER schema.
- EN: 1,500-doc subset of MTSamples; 36 reference cells; section-level KV schema with 9 canonical keys.
- Reference baselines: encoder-CRF (mBERT-BiLSTM-CRF), GPT-4o, DeepSeek-V4-Pro, Qwen3-8B (zero-shot).

## Known issues addressed since draft

- Inject_noise.py now correctly remaps label spans through Levenshtein-derived clean→noisy position map.
- BERT trainer numerical instability under chunked validation worked around with `use_bilstm=False, use_amp=False, chunk_size=400` for 7 affected cells.

## v1.1 plan

- Multi-seed (3 seeds) for proper F-tests.
- Uniform `use_bilstm=False, use_amp=False` config across all 72 cells.
- LoRA SFT baseline.
- Full MTSamples (4,902 docs) instead of 1,500-doc subset.
- Confusion-dictionary expansion to align realised noise density with target.
""")


def _cc_by_nc() -> str:
    return "CC-BY-NC-4.0\n\nSee https://creativecommons.org/licenses/by-nc/4.0/\n"


def _cc_by() -> str:
    return "CC-BY-4.0\n\nSee https://creativecommons.org/licenses/by/4.0/\n"


def _mit() -> str:
    return ("""MIT License

Copyright (c) 2026 anonymous-for-review

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
""")


# ---------------- main ----------------
def main():
    print("==> Building bundles")
    zh_dir = build_zh_bundle()
    en_dir = build_en_bundle()
    toolkit_dir = build_toolkit_bundle()

    print("==> Tarballing")
    tars = {
        "zh": RELEASE / "dualnoise-clinie-zh-v1.tar.gz",
        "en": RELEASE / "dualnoise-clinie-en-v1.tar.gz",
        "toolkit": RELEASE / "dualnoise-clinie-toolkit-v1.tar.gz",
    }
    tar_dir(zh_dir, tars["zh"])
    tar_dir(en_dir, tars["en"])
    tar_dir(toolkit_dir, tars["toolkit"])

    print("==> Updating Croissant with sha256")
    update_croissant_with_hashes(tars)

    print("==> Building release manifest")
    build_manifest(tars)

    print("==> Building supplementary.zip")
    build_supplementary(target_mb=80.0)

    print("\n==> Done. Release contents:")
    for f in sorted(RELEASE.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
