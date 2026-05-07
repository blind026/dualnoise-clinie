# DualNoise-ClinIE Toolkit (v1)

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
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 'tokenizers<0.20' peft==0.12.0 'numpy<2' \
    pytorch-crf safetensors python-Levenshtein
```

## Pipeline

```bash
python code/ccks_hf_to_clean.py --output data/raw/ccks2019/all.jsonl
python code/inject_noise.py --input data/processed/ccks_clean.jsonl \
    --output data/processed/ccks_noise/ --lang zh --levels 0,1,2,3,4,5 --seed 42
python code/annotation_error_sim.py --input data/processed/ccks_noise/L3.jsonl \
    --output data/processed/ccks_cells/L3_q0.6.jsonl --rho-a-r 0.6 --seed 42
python code/convert_to_labelstudio.py \
    --input data/processed/ccks_cells/L3_q0.6.jsonl \
    --output data/processed/ccks_ls/L3_q0.6.json --lang zh --schema flat
```

## License

MIT.
