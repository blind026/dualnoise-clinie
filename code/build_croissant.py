"""
Build Croissant 1.0 .jsonld metadata for the DualNoise-ClinIE v1 release.

Produces release/croissant_zh.jsonld and release/croissant_en.jsonld.

Usage:
  python build_croissant.py
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/Users/haiyang/海心/论文/ToBeOrNotToBe")

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def _fields_for_track(track: str) -> list:
    """Field schema with `source` pointers per Croissant 1.0 spec."""
    fileset_id = f"cells-{track}"
    prefix = "ccks_cell" if track == "zh" else "mts_cell"
    common = [
        ("id", "Document ID.", "sc:Text", "$.id"),
        ("text", "OCR-noisy text.", "sc:Text", "$.text"),
        ("labels", "List of {key, value, span, ...}.", "sc:Text", "$.labels"),
        ("clean_text_id", "Reference to clean source.", "sc:Text", "$.clean_text_id"),
        ("target_density", "Target density.", "sc:Float", "$.target_density"),
        ("realised_density", "Realised density.", "sc:Float", "$.realised_density"),
    ]
    out = []
    for fname, desc, dtype, jpath in common:
        out.append({
            "@type": "cr:Field",
            "@id": f"{prefix}/{fname}",
            "name": fname,
            "description": desc,
            "dataType": dtype,
            "source": {
                "fileSet": {"@id": fileset_id},
                "extract": {"jsonPath": jpath},
            },
        })
    return out


def base_metadata(track: str) -> dict:
    """Build the static metadata for one track (zh or en)."""
    if track == "zh":
        name = "DualNoise-ClinIE-zh"
        description = (
            "DualNoise-ClinIE Chinese release (v1). 8,157 source documents from CCKS 2019 Task 1 "
            "(via the doushabao4766/ccks_2019_ner_k_V3 HuggingFace mirror). Six entity classes (disease/diagnosis, "
            "anatomical site, imaging examination, lab examination, surgery, medication). 36 reference cells covering "
            "a 6 OCR-noise-level x 6 annotation-quality-level grid. v1 ZH track is flat 6-class entity NER. "
            "Released under CC-BY-NC-4.0 inheriting the upstream CCKS research-license terms."
        )
        license_url = "https://creativecommons.org/licenses/by-nc/4.0/"
        url = "https://dataverse.harvard.edu/previewurl.xhtml?token=9264631c-8774-41e7-b797-70e5cfb7a0ba"
        record_set_fields = _fields_for_track("zh")
    else:
        name = "DualNoise-ClinIE-en"
        description = (
            "DualNoise-ClinIE English release (v1). 1,500-document subset of MTSamples (full 4,902-doc set "
            "reproducible from the toolkit). Silver section-level KV labels covering nine canonical schema keys "
            "(chief_complaint, history_of_present_illness, past_medical_history, social_family_history, "
            "medications_allergies, physical_examination, diagnostic_studies, assessment_diagnosis, plan_treatment). "
            "36 reference cells (6 OCR-noise levels x 6 annotation-quality levels). v1 EN track is section-level KV. "
            "Released under CC-BY-4.0 consistent with upstream MTSamples licensing."
        )
        license_url = "https://creativecommons.org/licenses/by/4.0/"
        url = "https://dataverse.harvard.edu/previewurl.xhtml?token=9264631c-8774-41e7-b797-70e5cfb7a0ba"
        record_set_fields = _fields_for_track("en")

    md = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "name": name,
        "description": description,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "version": "1.0.0",
        "citeAs": "@inproceedings{dualnoise2026, title={DualNoise-ClinIE: A Calibrated Benchmark for Joint OCR and Annotation Noise in Semi-Structured Clinical Information Extraction}, author={anonymous}, booktitle={NeurIPS 2026 Datasets and Benchmarks Track (under review)}, year={2026}}",
        "license": license_url,
        "url": url,
        "creator": {"@type": "sc:Organization", "name": "anonymous-for-review"},
        "publisher": {"@type": "sc:Organization", "name": "anonymous-for-review"},
        "datePublished": "2026-05-07",
        "keywords": [
            "clinical NLP", "information extraction", "OCR noise",
            "annotation noise", "benchmark", "Chinese clinical NER",
            "English MTSamples", "section-level KV extraction"
        ],
        "isLiveDataset": False,
        # Files. The release tarball is the only top-level FileObject (with sha256);
        # individual JSONL files inside it are described as FileSets via containedIn.
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": f"release-archive-{track}",
                "name": f"release-archive-{track}",
                "description": (
                    f"Top-level release archive (gzipped tar) containing the {track.upper()} clean source corpus and the "
                    "36 reference cells. SHA256 is computed at release time and pinned here."
                ),
                "contentUrl": f"dualnoise-clinie-{track}-v1.tar.gz",
                "encodingFormat": "application/gzip",
                "sha256": "PLACEHOLDER-archive-sha256",
            },
            {
                "@type": "cr:FileSet",
                "@id": f"clean-source-{track}",
                "name": f"clean-source-{track}",
                "description": "Clean source documents (pre-noise) inside the release archive, used to derive the noisy cells.",
                "containedIn": {"@id": f"release-archive-{track}"},
                "encodingFormat": "application/jsonlines",
                "includes": ("clean/ccks_clean.jsonl" if track == "zh" else "clean/mtsamples_clean_1500.jsonl"),
            },
            {
                "@type": "cr:FileSet",
                "@id": f"cells-{track}",
                "name": f"cells-{track}",
                "description": "36 reference cells (6 OCR-noise levels L0..L5 x 6 annotation-quality levels q0.0..q1.0) inside the release archive.",
                "containedIn": {"@id": f"release-archive-{track}"},
                "encodingFormat": "application/jsonlines",
                "includes": "cells/L*_q*.jsonl",
            },
        ],
        # Logical record set: one record per document per cell
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": f"{track}-cells-records",
                "name": f"{track}-cells-records",
                "description": "One record per (document, OCR-level, annotation-quality-level).",
                "field": record_set_fields,
            }
        ],
    }

    # NeurIPS Responsible AI (RAI) metadata — full set per NeurIPS 2026 D&B guidelines
    rai_zh = (track == "zh")
    md["rai:dataCollection"] = (
        "Two-stage collection. (1) Source-corpus stage. ZH: the 8,157-document CCKS 2019 Task 1 corpus, mirrored at "
        "huggingface.co/datasets/doushabao4766/ccks_2019_ner_k_V3. EN: the ~5,000-document MTSamples corpus of "
        "transcribed English clinical reports, mirrored at huggingface.co/datasets/harishnair04/mtsamples. Both upstream "
        "corpora are publicly available and de-identified by their original maintainers. (2) Calibration stage (statistics-only). "
        "OCR-noise and annotation-error parameters were fitted from an in-house oncology corpus of 1,900+ paper-form "
        "clinical reports captured at point of care, processed by a privately deployed OCR engine, and annotated by a "
        "24-person team. The calibration corpus itself is NOT redistributed; only the aggregated noise-statistics parameters "
        "(Gamma fits, taxonomy proportions, Beta-mixture annotator parameters) are released as part of this benchmark."
    )
    md["rai:dataCollectionType"] = (
        "Re-distribution of upstream public corpora plus synthetic noise/error injection calibrated against an in-house corpus."
    )
    md["rai:dataCollectionRawData"] = (
        "Upstream raw corpora are NOT modified; we redistribute only the noise-perturbed derivatives "
        "(36 cells per language) plus pointers to the upstream sources for the clean text."
    )
    md["rai:dataCollectionTimeframe"] = (
        "Upstream CCKS 2019 originally released 2019; MTSamples is a stable static collection from the early 2010s. "
        "Calibration corpus collected 2024-04 to 2025-12 under an institutional patient-management programme. "
        "v1 derivative artifacts generated 2026-05-06 to 2026-05-07."
    )
    md["rai:dataImputationProtocol"] = "No imputation. Missing labels (omitted entities) are treated as absent during scoring."
    md["rai:dataPreprocessingProtocol"] = (
        "Five-stage deterministic pipeline (seeded by --seed): "
        "(i) source-corpus loading and de-duplication; "
        "(ii) for EN, silver section-level labelling via regex section-header parser mapping detected headers "
        "(CHIEF COMPLAINT:, PAST MEDICAL HISTORY:, SUBJECTIVE:, ASSESSMENT:, PLAN:, etc.) to nine canonical schema keys; "
        "(iii) OCR-noise injection across an 8-category (ZH) or 7-category (EN) taxonomy at six reference density levels, "
        "calibrated against the deployed-corpus Gamma fit; "
        "(iv) annotation-error simulation via a two-component Beta mixture (senior/junior annotator subgroups) at six "
        "reference senior-annotator-ratio levels, partitioning corruptions between omission and boundary shift; "
        "(v) Levenshtein-opcode-based label-span remapping from clean to noisy coordinates so that all annotations remain "
        "valid in the noisy text."
    )
    md["rai:dataAnnotationProtocol"] = (
        "ZH labels are inherited unchanged from the CCKS 2019 Task 1 gold annotations (six entity classes annotated by "
        "domain experts at the original release). "
        "EN labels are silver-quality, generated automatically by a regex section-header parser; the parser was tuned on "
        "manual review of 100 randomly sampled MTSamples documents. The 200-document subset of MTSamples used as the "
        "evaluation surface had the silver labels manually inspected."
    )
    md["rai:dataAnnotationPlatform"] = "Internal Python toolchain; no third-party annotation platform involved for v1."
    md["rai:dataAnnotationAnalysis"] = (
        "Inter-annotator agreement applies only to the calibration corpus (24 annotators, K-Means K=2 cluster analysis "
        "yielded senior/junior subgroups with mean error rates 6.4% and 35.1% respectively). "
        "ZH gold labels: not re-annotated by us. EN silver labels: not subject to inter-annotator agreement (single regex parser)."
    )
    md["rai:dataAnnotationPerItem"] = (
        "ZH: 6 entity classes (disease_diagnosis, anatomical_site, imaging_exam, lab_exam, surgery, medication). "
        "EN: 9 section keys (chief_complaint, history_of_present_illness, past_medical_history, social_family_history, "
        "medications_allergies, physical_examination, diagnostic_studies, assessment_diagnosis, plan_treatment). "
        "Average labels per document: ZH ~2.4; EN ~4.5."
    )
    md["rai:dataAnnotationDemographics"] = (
        "Calibration corpus: 24 annotators, demographic information not collected, compensated at or above local prevailing rates. "
        "Upstream CCKS / MTSamples annotator demographics: not disclosed by upstream releases."
    )
    md["rai:dataAnnotationTools"] = (
        "Toolkit (released MIT-licensed): code/inject_noise.py, code/annotation_error_sim.py, code/silver_label.py, "
        "code/convert_to_labelstudio.py, code/eval_cell.py."
    )
    md["rai:dataReleaseMaintenancePlan"] = (
        "Maintained for at least 24 months by the authors. Erratum versioned in CHANGELOG.md and ERRATA.md. "
        "Versioned releases pre-registered: v1.1 (multi-seed, LoRA SFT, uniform config, full MTSamples set, expanded "
        "confusion dictionary), v1.2 (broader entity schema), v2.0 (MIMIC-IV-Note via PhysioNet credentialed channel), "
        "v3.0 (vision-language baselines that bypass OCR)."
    )
    md["rai:dataSocialImpact"] = (
        "Intended use: research on noise-robust clinical information extraction. Beneficiaries: clinical-NLP researchers, "
        "noise-aware-training methodology researchers. Potential negative use: results obtained on this benchmark do NOT "
        "support deployment of clinical IE systems for real patient care without separate prospective clinical validation. "
        "We explicitly disclaim use for diagnostic, treatment, triage, or regulatory submission decisions."
    )
    md["rai:dataBiases"] = (
        "Calibration corpus reflects a single Chinese oncology setting; the Beta-mixture annotator-error model is fit to "
        "that team's behaviour and may not generalise to teams with different training or quality-control regimes. "
        "ZH source corpus reflects CCKS 2019 contributing institutions and is biased toward inpatient/oncology genres. "
        "EN source corpus (MTSamples) is enriched in dictation transcripts and uneven across specialties. "
        "v1 EN release uses a 1,500-document subset for tractability; specialty distribution may differ from the full set."
    )
    md["rai:dataLimitations"] = (
        "(1) ZH track is flat 6-class entity NER on sentence-fragments (the HF mirror does not preserve CCKS section structure); "
        "the originally-targeted header-anchored KV framing is downgraded for v1 ZH. "
        "(2) EN labels are silver-quality (regex section detection); zero-shot LLM evaluation is structurally suppressed by "
        "format mismatch with silver-label tokenisation. "
        "(3) v1 uses a 1,500-doc EN subset (4,902-doc full set is reproducible from the toolkit, slow under v1's pure-Python "
        "position-remapping). "
        "(4) 1 seed only; v1.1 will report multi-seed runs with proper F-tests. "
        "(5) ZH grid mixes BiLSTM-on (31 cells) and BiLSTM-off (5 workaround cells); EN grid is uniform `use_bilstm=False`. "
        "(6) Realised noise density on the v1 release is ~37% of target due to per-category injector eligibility preconditions; "
        "v1.1 will expand the confusion dictionary."
    )
    md["rai:personalSensitiveInformation"] = (
        "Both upstream corpora (CCKS 2019, MTSamples) are released by their original maintainers in de-identified form, "
        "and we do not re-introduce identifiable information at any stage of v1 construction. "
        "The private calibration corpus is processed under documented institutional patient-protection protocols and is NOT "
        "redistributed in any form; only aggregated noise-statistics parameters are released. "
        "v1 release files contain no patient names, MRNs, dates of birth, addresses, or other PII fields."
    )
    md["rai:dataUseCases"] = (
        "Recommended uses: (i) evaluation of noise-aware training methods for clinical IE; (ii) study of cost-quality trade-offs "
        "in annotation workflows by sweeping the senior-annotator ratio; (iii) cross-family comparison of extraction model "
        "families (encoder-CRF, open-weight LLMs, frontier closed LLMs) under shared evaluation conditions; (iv) benchmarking "
        "of post-OCR correction methods; (v) development of language-aware OCR noise models. "
        "Out-of-scope: clinical diagnosis, triage, treatment selection, regulatory submission, or real-patient deployment."
    )
    md["rai:hasSyntheticData"] = (
        "Yes (mixed). The text spans of every cell are derived from the public upstream corpora "
        f"({'CCKS 2019 Task 1 HuggingFace mirror' if track == 'zh' else 'MTSamples'}) by applying a synthetic OCR-noise "
        "injection pipeline (8 categories ZH / 7 categories EN, 6 reference density levels). The annotation perturbations "
        "(omissions and boundary shifts) are generated by a synthetic two-component Beta-mixture annotator-error simulator. "
        "Both injectors are calibrated against an in-house deployed-corpus statistics. "
        "The CLEAN source text is unmodified from the upstream corpus; the noisy variants and label perturbations are synthetic."
    )
    md["rai:wasDerivedFrom"] = [
        {
            "@type": "sc:Dataset",
            "name": ("CCKS 2019 Task 1 (HF mirror: doushabao4766/ccks_2019_ner_k_V3)" if track == "zh"
                     else "MTSamples (HF mirror: harishnair04/mtsamples)"),
            "url": ("https://huggingface.co/datasets/doushabao4766/ccks_2019_ner_k_V3" if track == "zh"
                    else "https://huggingface.co/datasets/harishnair04/mtsamples"),
            "description": (
                "Upstream public corpus from which the clean source text and (for ZH) gold entity annotations are inherited. "
                "We do not redistribute the upstream corpus; only the noise-perturbed derivatives plus pointers to the "
                "upstream sources are released."
            ),
        },
        {
            "@type": "sc:Dataset",
            "name": "Internal calibration corpus (oncology clinical reports, 24 annotators)",
            "url": "private; statistics-only release in this benchmark",
            "description": (
                "An in-house corpus of 1,900+ paper-form oncology clinical reports captured at point of care, processed by "
                "a privately deployed OCR engine, and annotated by a 24-person team. Used SOLELY to fit the noise-injection "
                "and annotation-error simulator parameters (Gamma fits, taxonomy proportions, Beta-mixture annotator parameters). "
                "The corpus itself is NOT redistributed; only the fitted parameters are."
            ),
        },
    ]
    md["rai:wasGeneratedBy"] = [
        {
            "@type": "sc:SoftwareApplication",
            "name": "DualNoise-ClinIE toolkit (v1)",
            "url": "https://anonymous.4open.science/r/<TBD>/",
            "description": (
                "The open-source toolkit that generated this dataset. Pipeline (deterministic given seed): "
                "(i) source-corpus loading; "
                "(ii) for EN, silver section-level labelling via regex section-header parser; "
                "(iii) OCR-noise injection (8-category ZH / 7-category EN, 6 density levels, calibrated against the in-house corpus); "
                "(iv) annotation-error simulation (Beta-mixture, 6 senior-annotator-ratio levels); "
                "(v) Levenshtein-opcode-based label-span remapping from clean to noisy coordinates. "
                "Toolkit released under MIT license."
            ),
            "softwareVersion": "1.0.0",
            "license": "https://opensource.org/licenses/MIT",
        }
    ]

    return md


def main():
    out_dir = ROOT / "release"
    out_dir.mkdir(exist_ok=True)
    for track in ("zh", "en"):
        md = base_metadata(track)
        path = out_dir / f"croissant_{track}.jsonld"
        path.write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size} bytes)")

        # Validate via mlcroissant
        try:
            import mlcroissant as mc
            ds = mc.Dataset(jsonld=str(path))
            print(f"  ✓ validates OK with mlcroissant")
        except Exception as e:
            print(f"  ! mlcroissant validation warning: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
