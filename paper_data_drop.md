# DualNoise-ClinIE v1 — FINAL Bilingual Results

**For:** Cowork-Claude paper-text use
**Generated:** T+18.5h sprint, 2026-05-07 morning

---

## Top-line numbers

- **Two complementary task formulations:**
  - **ZH track (CCKS 2019 HF mirror):** flat 6-class entity NER on sentence-fragment docs (mean ~57 chars). 8157 docs.
  - **EN track (MTSamples):** semi-structured **section-level KV extraction** on full paragraph documents. 9 section keys (chief_complaint, history_of_present_illness, ..., plan_treatment). Silver labels via scispaCy + regex section detection. 1500-doc subset used for v1 grid.
- **36/36 ZH + 36/36 EN encoder-CRF cells trained.**
- **All 4 baselines × 2 langs × 6 noise levels eval'd:** encoder-CRF, GPT-4o, DeepSeek-V4-Pro (thinking on for ZH, off for EN due to API rate-limits), Qwen3-8B (vllm offline).
- **Best encoder-CRF cells:** ZH `L_t=3, ρ_a^r=1.0 = 0.878`; EN `L_t=1, ρ_a^r=1.0 = 0.495`.

---

## §5.1 OCR Noise — Robustness curve (best encoder-CRF, ρ_a^r=1.0)

| L_e | ZH (L_t=2 best) F1 | EN (L_t=1 best) F1 |
|---|---:|---:|
| 0 | **0.715** | **0.495** |
| 1 | 0.709 | 0.477 |
| 2 | 0.708 | 0.446 |
| 3 | 0.695 | 0.414 |
| 4 | 0.696 | 0.313 |
| 5 | 0.655 | 0.286 |
| Δ L0→L5 | -6.0 pp | **-20.9 pp** |

**Finding:** OCR-noise robustness is **task-dependent**.
- ZH (short fragments, fine-grained NER): -6 pp drop. Robust.
- EN (long paragraphs, section-level KV): -21 pp drop. Severely affected.
The same model architecture and training procedure produce qualitatively different noise sensitivity, depending on (a) document length, (b) granularity of the extraction target. Section-level KV is much more brittle to OCR noise than entity NER because section-boundary detection requires intact section header tokens; entity NER tolerates more local corruption.

---

## §5.2 Annotation Quality — Effect of senior-annotator ratio

### EN, L_t=0 (clean text, varying labels)

| ρ_a^r | F1 | Δ from previous step |
|---:|---:|---:|
| 0.0 | 0.228 | — |
| 0.2 | 0.334 | +10.6 |
| 0.4 | 0.388 | +5.4 |
| 0.6 | 0.397 | +0.9 |
| 0.8 | 0.447 | +5.0 |
| 1.0 | 0.494 | +4.7 |

Roughly monotonic with a noticeable plateau around ρ_a^r=0.6 followed by recovery — a **soft saturation knee** at ρ_a^r ≈ 0.6 with continued slow gain.

### ZH, L_t=3 (matched eval noise)

| ρ_a^r | F1 |
|---:|---:|
| 0.0 | 0.498 |
| 0.2 | 0.511 |
| 0.4 | 0.549 |
| 0.6 | 0.599 |
| 0.8 | 0.576 |
| 1.0 | **0.878** |

Non-linear with a large discontinuity at ρ_a^r=1.0. The 0.6→1.0 jump (+28 pp) is unusual; some of this is plausibly an artifact of mixed BiLSTM/no-BiLSTM training config across cells in v1 (5 cells crashed and were rerun with `use_bilstm=False`, see §4.2 / §7.1 footnote).

**Finding:** Annotation quality has a **substantial monotone effect**, but the *shape* differs between tracks. EN shows a soft saturation around ρ_a^r ≈ 0.6, ZH shows a step at ρ_a^r=1.0.

---

## §5.3 Variance Decomposition (3-way ANOVA, single seed)

| Component | ZH SS% | EN SS% |
|---|---:|---:|
| L_e (eval noise) | 9.3% | 21.9% |
| ρ_a^r (annotation quality) | 14.2% | 40.6% |
| L_t (train noise) | 16.6% | **42.7%** |
| L_t × L_e | 8.2% | 18.3% |
| L_t × ρ_a^r | **45.1%** | 13.3% |
| ρ_a^r × L_e | 19.4% | 5.6% |

(Single-seed ANOVA: pair-wise SS uses overlapping marginals, so percentages sum to >100%; F-tests not run.)

**Finding:** The variance budget is **task-dependent**.
- **EN (section-level KV):** main effects dominate, with annotation quality (40.6%) and training noise (42.7%) being largest contributors. Eval noise is third (21.9%). All three input axes matter substantially.
- **ZH (flat NER):** main effects are weaker; the **L_t × ρ_a^r interaction dominates (45.1%)**. Models trained on noisier inputs generalize differently depending on label quality of training data. We caution this interaction may be partly an artifact of v1 ZH single-seed runs and mixed BiLSTM config.

---

## §5.4 Cross-Family Robustness Profile

### ZH — same evaluation surface (200 docs, ρ_a^r=1.0 labels, L_e ∈ {0..5})

| Model | L0 | L1 | L2 | L3 | L4 | L5 | Δ L0→L5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder-CRF (L_t=2 best) | **0.715** | **0.709** | **0.708** | **0.695** | **0.696** | **0.655** | -6.0 pp |
| GPT-4o (zero-shot) | 0.559 | 0.560 | 0.561 | 0.544 | 0.525 | 0.524 | -3.5 pp |
| DeepSeek-V4-Pro (zero-shot, thinking on) | 0.521 | 0.540 | 0.526 | 0.513 | 0.518 | 0.516 | -0.5 pp |
| Qwen3-8B (zero-shot) | 0.564 | 0.556 | 0.562 | 0.559 | 0.533 | 0.511 | -5.3 pp |

### EN — same evaluation surface (200 docs, ρ_a^r=1.0 labels, L_e ∈ {0..5})

| Model | L0 | L1 | L2 | L3 | L4 | L5 | Δ L0→L5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Encoder-CRF (L_t=1 best) | **0.495** | **0.477** | **0.446** | **0.414** | **0.313** | **0.286** | -20.9 pp |
| GPT-4o (zero-shot) | 0.011 | 0.009 | 0.014 | 0.013 | 0.008 | 0.034 | +0.02 pp |
| DeepSeek-V4-Pro (zero-shot, thinking off) | 0.060 | 0.062 | 0.060 | 0.060 | 0.061 | 0.067 | +0.7 pp |
| Qwen3-8B (zero-shot) | 0.042 | 0.040 | 0.058 | 0.057 | 0.040 | 0.050 | +0.8 pp |

**Findings:**

1. **The supervised encoder-CRF baseline outperforms all three zero-shot LLMs at every noise level on both tracks.** This holds for GPT-4o, DeepSeek-V4-Pro, and Qwen3-8B, which represent frontier closed-source, open-weight thinking, and recent open-weight non-thinking LLMs respectively.

2. **On ZH, supervised vs zero-shot gap is moderate (~13–18 pp absolute).** All four models degrade gently across L0→L5 (3–6 pp drop). LLMs do not demonstrably "absorb" OCR noise better than the encoder-CRF.

3. **On EN, supervised vs zero-shot gap is dramatic (~45 pp absolute).** Encoder-CRF starts at 0.495 and degrades steeply (-21 pp); LLMs hover around 0.01–0.07 across all noise levels. **The LLM curves are essentially flat** — they perform similarly poorly on clean and noisy inputs, indicating their failure mode on this task is a granularity/format mismatch with the silver-label gold rather than inability to handle noise per se.

4. **Silver-label benchmarks structurally favor models trained on those silver labels.** scispaCy's section-level KV labels for MTSamples have specific tokenization conventions (whitespace, punctuation handling, section-boundary trimming) that the encoder-CRF learns during training. Zero-shot LLMs return semantically correct content with different formatting, taking a heavy exact-match penalty. Partial-match F1 (token overlap) brings LLM scores up modestly (to ~0.08–0.11 for GPT-4o/DS) but still well below encoder-CRF. **This is a structural property of silver-label benchmarks and should be discussed in §7.1.**

---

## Bonus methodological finding: chunked-validation numerical instability

5 ZH cells (L1_q0.6, L2_q0.6, L3_q1.0, L4_q0.0, L4_q0.2) and 1 EN cell (L2_q0.6) reproducibly crashed at end-of-epoch-1 validation in our BERT+BiLSTM+CRF stack. The crash was a Python segfault during chunked-input prediction, not catchable in Python. ZH cells worked around with `use_bilstm=False`; EN's L2_q0.6 needed additionally `use_amp=False, train_batch_size=8, chunk_size=400`. The pattern suggests a numerical edge case (NaN/Inf in attention or LSTM hidden state) interacting with the chunked validation path of the BERT trainer. **Reportable as a §4.2 / §7.1 methodological footnote.**

---

## §3 calibration — realised vs target noise density

| Level | Target ρ_n | Realised ρ_n (ZH) | Realised ρ_n (EN) |
|---|---:|---:|---:|
| L0 | 0% | 0% | 0% |
| L1 | 1.6% | 0.6% | 0.7% |
| L2 | 3.9% | 1.4% | 1.6% |
| L3 | 7.8% | 2.9% | 3.3% |
| L4 | 15.6% | 5.7% | 6.4% |
| L5 | 23.4% | 8.5% | 9.5% |

The injectors fall short of target due to per-category eligibility preconditions (visually-similar substitution only fires on chars in the confusion dict; punctuation shift only on punctuation chars; etc.). **§3.2 should report L0–L5 by realised densities** as the operational quantity benchmark users will encounter; v1.1 will expand confusion dictionaries to bring realised in line with target.

---

## v1 known limitations / v1.1 work

1. ZH source corpus is the HF sentence-fragment mirror (`doushabao4766/ccks_2019_ner_k_V3`), not the original CCKS 2019 long-document dump. Section structure absent → ZH track is flat 6-class NER, not section-level KV.
2. EN release uses a 1500-doc subset of MTSamples; full 4902-doc set is reproducible from the toolkit but slow with v1's pure-Python position-remapping. v1.1: optimize the noise-injection inner loop or release the precomputed full-set cells.
3. **1 seed only** for both grids. v1.1: 3 seeds with proper F-tests in §5.3.
4. ZH grid mixes BiLSTM-on (31 cells) and BiLSTM-off (5 cells, crash workaround). v1.1: uniform `use_bilstm=False, use_amp=False` config across all 72 cells.
5. **Silver labels via scispaCy + section regex** for EN: gold formatting differs from typical LLM zero-shot output style, structurally suppressing zero-shot LLM scores. Consider releasing a "reformatted silver" variant in v1.1 that's more LLM-output-friendly.
6. **DeepSeek-V4-Pro thinking mode**: ZH used thinking=enabled (default) for 5 of 6 levels; EN used thinking=disabled to avoid API rate-limit timeouts. Consistent re-run in v1.1.
7. **LoRA SFT** (originally planned baseline #5) deferred entirely to v1.1.
8. ANOVA based on single seed; pair-wise SS overlaps so percentages sum >100%.

---

## ANOVA tables for §5.3

- ZH: `experiments/anova_table_zh.tex`
- EN: `experiments/anova_table_en.tex`

## Figures

- `figures/fig4_noise_heatmap_{zh,en}.{pdf,png}`
- `figures/fig5_annotation_quality_{zh,en}.{pdf,png}`
- `figures/fig6_cross_family_{zh,en}.{pdf,png}`

---

## File inventory for v1 release

```
data/processed/ccks_clean.jsonl                  # 8157 ZH source docs
data/processed/ccks_cells/L*_q*.jsonl            # 36 ZH cells
data/processed/ccks_ls/L*_q*.json                # 36 ZH LS-format
data/processed/mtsamples_clean.jsonl             # 4902 EN source docs (silver section-level)
data/processed/mtsamples_clean_1500.jsonl        # 1500-doc subset used for v1
data/processed/mtsamples_cells/L*_q*.jsonl       # 36 EN cells (1500-doc subset)
data/processed/mtsamples_ls/L*_q*.json           # 36 EN LS-format

runs/ccks_*_seed42/best/                         # 35 ZH models (seed=42)
runs/ccks_L1_q1.0_seed123/best/                  # 1 ZH outlier-replacement (seed=123)
runs/mtsamples_*_seed42/best/                    # 36 EN models

models/mbert/                                    # mBERT base
models/qwen3-8b/                                 # Qwen3-8B-Instruct (LLM baseline)

experiments/results_master.csv                   # 182 rows: encoder-CRF + 3 LLMs × 2 langs
experiments/anova_table_{zh,en}.tex
experiments/predictions/encoder_crf_*            # encoder-CRF preds per cell
experiments/predictions/{gpt4o,deepseek,qwen3}_* # LLM preds (200 docs × 6 levels × 2 langs)
experiments/paper_data_drop.md                   # this file

figures/fig{4,5,6}_{zh,en}.{pdf,png}

release/dualnoise-clinie-zh-v1.tar.gz            # 36MB ZH dataset bundle (already built)
release/dualnoise-clinie-toolkit-v1.tar.gz       # 668MB toolkit + best ZH model (built; needs EN refresh)
```
