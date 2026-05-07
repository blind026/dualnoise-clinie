# DualNoise-ClinIE v1 — Release Notes & Pivots from Original Plan

**For:** Cowork-Claude / paper-update use  
**Updated:** 2026-05-06

---

## What changed from the original plan

### 1. ZH-only release (EN deferred to v1.1)

**Original plan:** Bilingual release (CCKS 2019 Chinese + MTSamples English).  
**Actual v1:** ZH-only.

**Reasons:**
- scispaCy `en_core_sci_md` install blocked: AWS S3 (us-west-2) timeouts from China network; HuggingFace mirror version is gated; alternative public HF repos (`jdallain/en_core_sci_sm`, etc.) are empty.
- Per the 3-day plan's hard triage tree, "EN release moves to Appendix-only / future work" is drop step #3.

**Paper edits required:**
- Abstract: change "bilingual coverage" → "Chinese release in v1; English release planned for v1.1"
- §1: same
- §3.1: keep MTSamples description but flag "v1.1" status
- §4: encoder-CRF and LLM baselines on ZH only
- §5: ZH-only analyses; the §5.5 cross-language paragraph drops to "deferred to v1.1"
- §7.1 Limitations: explicit note about ZH-only v1 + planned EN

### 2. CCKS source switched from full-document to sentence-fragment HF mirror

**Original plan:** CCKS 2019 Task 1 official corpus (~1600 full inpatient reports with section structure 主诉/现病史/.../诊断).  
**Actual v1:** `doushabao4766/ccks_2019_ner_k_V3` HuggingFace mirror — 8157 sentence-fragment docs (mean ~57 chars), entity-tagged but with no preserved section structure.

**Reasons:**
- The original CCKS 2019 archive is not trivially redistributable (no clean public download).
- The HF mirror provides BIO-tagged sentences derived from CCKS 2019 with the same 6 entity types.

**Implications:**
- Section headers absent in source → "semi-structured KV via section anchors" framing **does not apply** to ZH track in v1.
- The ZH track is repositioned as **flat NER over 6 entity classes** (disease_diagnosis, anatomical_site, imaging_exam, lab_exam, surgery, medication).
- The §3 framing changes: instead of "field-header restoration → KV format", the ZH pipeline emits each entity directly with its type as the label.

**Paper edits required:**
- §3.1: rewrite the CCKS paragraph to acknowledge source = HF sentence-fragment mirror; defend "still real OCR-target data because each fragment is a real clinician utterance"
- §3.5: "the recommended primary metric is exact-match F1 over (entity_type, span) pairs" instead of (key, value) pairs
- §4: encoder-CRF predicts BIOE labels over 6 entity types; KV-pairing post-processing removed for ZH
- The Datasheet's CCKS source URL/citation should point to both the official CCKS 2019 release and this HF mirror

### 3. Noise calibration realises lower density than target

**Plan:** Reference levels at 0%, 1.6%, 3.9%, 7.8%, 15.6%, 23.4% (within ±0.5pp tolerance).  
**Actual:** L0=0%, L1=0.6%, L2=1.4%, L3=2.9%, L4=5.7%, L5=8.5%.

**Reason:** The eight-category injectors have strong eligibility preconditions (e.g., visually-similar substitution only fires on chars in the confusion dict, which is currently a 4-entry stub). The expected per-position probability ρ_n × α_j is computed but only fires for a fraction of eligible positions. The expected-density model assumes all positions are eligible for all categories, which isn't realised.

**Paper edits required (one of three options):**
- **Option A:** Re-derive level multipliers based on realised density. E.g., set the multipliers so realised L3 ≈ 7.8%. This requires re-injecting and re-training all 36 cells, adding ~2 hours.
- **Option B:** Replace the target-density framing in §3.2 with realised-density framing: "L_ℓ corresponds to realised density ρ_n^{(ℓ)} as measured on the calibration corpus" with the empirical numbers.
- **Option C (chosen for v1 due to time):** Document the discrepancy in §3.3 ("realised density falls short of target due to per-category eligibility constraints; the qualitative noise gradient is preserved and §5.1's training-evaluation alignment effect remains observable; the toolkit's confusion dictionary will be expanded in v1.1 to bring realised density into alignment").

### 4. Encoder-CRF grid: 1 seed (not 3)

**Plan:** 3 seeds × 36 cells = 108 runs.  
**Actual v1:** 1 seed × 36 cells = 36 runs (saving ~6 GPU-hours).

**Reasons:** Time pressure; standard deviation across seeds is a v1.1 polish item.

**Paper edits required:**
- §4.2: "3 seeds" → "1 seed in v1; 3 seeds planned for v1.1"
- §5: drop ± standard deviation reporting
- §5.3 ANOVA: report SS percentages without F-tests (no residual to estimate without seeds)

### 5. Bug fix to noise injection (label span remapping)

The original `code/inject_noise.py` did **not** update entity span positions after text mutations (insertions, deletions, rearrangements), making labels useless for any noise level > 0. Patched with Levenshtein-opcode-based clean→noisy position mapping (see `experiments/blocker.md`).

This is a behind-the-scenes fix and does not affect the paper, but the toolkit release should highlight it in the changelog.

### 6. Patch to BERT repo's `prepare_data.py`

Hardcoded `["键名", "值", "医院名称"]` filter in `has_valid_annotations` was preventing flat-NER labels from being accepted. Patched to accept any non-empty label set.

---

## Going forward — what to keep tracking for v1.1

1. Full bilingual release (English silver labels via downloadable scispaCy model)
2. Real CCKS 2019 full-document source (not sentence-fragment mirror)
3. Confusion dictionary expansion to bring noise calibration into target range
4. 3-seed runs for proper ANOVA F-tests
5. LoRA SFT variant (currently dropped if time tight)
6. Vision-language baselines (already explicitly v3.0)
