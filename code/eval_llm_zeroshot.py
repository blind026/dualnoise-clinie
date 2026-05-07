"""
DualNoise-ClinIE — LLM Zero-Shot Evaluation
============================================
Shared protocol for GPT-4o, DeepSeek-V4, and Qwen2.5-7B (via vLLM).

Usage
-----
python eval_llm_zeroshot.py \
    --model gpt-4o-2024-08-06 \
    --provider openai \
    --input data/processed/ccks_cells/L3_q0.6.jsonl \
    --gold  data/processed/ccks_gold_eval.jsonl \
    --lang  zh \
    --shuffle-seed 0 \
    --output experiments/predictions/gpt4o_zh_L3_s0.json
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Dict, List

PROMPT_SYSTEM_ZH = """你是一名临床信息抽取专家，从扫描-OCR 转录的中文病历文本中按字段抽取关键信息。
严格遵循以下规则：
1. 只抽取输入文本中逐字出现的实体；禁止改写、释义、合成或归纳。
2. 不进行任何医学推断或诊断扩展；不要补充输入中未明确出现的内容。
3. 输入文本可能含有 OCR 噪声（错字、多余空格、视觉相似字符替换等），按文本字面抽取，不要纠正噪声。
4. 输出必须严格符合下方提供的 JSON Schema；任何字段若未在输入中出现，对应数组留空（[]）。
5. 同一字段下的多个实体按其在原文中出现的顺序排列。"""

PROMPT_SYSTEM_EN = """You are a clinical information extraction expert. You extract key information from
scanned-OCR transcribed English clinical reports, organised by section header.
Strictly follow these rules:
1. Extract only entities that appear verbatim in the input text; never paraphrase, summarise, infer, or invent.
2. Do not perform medical reasoning or diagnostic extension.
3. The input may contain OCR noise; extract verbatim, do not correct.
4. Output must strictly conform to the JSON schema; if a field is not present, return an empty array [].
5. Multiple entities within a single field are listed in the order they appear in the source text."""

SCHEMA_ZH = {
    "type": "object", "additionalProperties": False,
    "required": ["主诉", "现病史", "既往史", "查体", "辅助检查", "诊断", "治疗经过"],
    "properties": {k: {"type": "array", "items": {"type": "string"}} for k in
                   ["主诉", "现病史", "既往史", "查体", "辅助检查", "诊断", "治疗经过"]},
}
SCHEMA_EN = {
    "type": "object", "additionalProperties": False,
    "required": ["chief_complaint","history_of_present_illness","past_medical_history",
                 "physical_examination","diagnostic_studies","assessment_diagnosis","plan_treatment"],
    "properties": {k: {"type": "array", "items": {"type": "string"}} for k in
                   ["chief_complaint","history_of_present_illness","past_medical_history",
                    "physical_examination","diagnostic_studies","assessment_diagnosis","plan_treatment"]},
}

def build_prompt(report_text: str, lang: str, icl_examples: List[Dict]) -> List[Dict]:
    system = PROMPT_SYSTEM_ZH if lang == "zh" else PROMPT_SYSTEM_EN
    schema = SCHEMA_ZH if lang == "zh" else SCHEMA_EN
    msgs = [{"role": "system", "content": system}]
    msgs.append({"role": "system", "content": f"OUTPUT SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"})
    for ex in icl_examples:
        msgs.append({"role": "user", "content": f"病历文本:\n{ex['text']}" if lang=='zh' else f"REPORT TEXT:\n{ex['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["labels_json"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"病历文本:\n{report_text}" if lang=='zh' else f"REPORT TEXT:\n{report_text}"})
    return msgs

def to_predictions_format(model_output: Dict) -> List[Dict]:
    """Convert {key: [val, val, ...]} dict to [{key, value}] list."""
    out = []
    if not isinstance(model_output, dict): return out
    for k, vs in model_output.items():
        if not isinstance(vs, list): continue
        for v in vs:
            if isinstance(v, str) and v.strip():
                out.append({"key": k, "value": v.strip(), "span": [-1, -1]})
    return out

def call_openai(model: str, messages: List[Dict], schema: Dict) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=2048,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "extraction", "schema": schema, "strict": False}}
    )
    return resp.choices[0].message.content

def call_deepseek(model: str, messages: List[Dict]) -> str:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=2048,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def call_vllm(endpoint: str, model: str, messages: List[Dict]) -> str:
    import requests
    r = requests.post(f"{endpoint}/v1/chat/completions", json={
        "model": model, "messages": messages, "temperature": 0, "max_tokens": 2048,
    }, timeout=120)
    return r.json()["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["openai", "deepseek", "vllm"])
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--lang", required=True, choices=["zh", "en"])
    ap.add_argument("--icl", type=Path, default=None,
                    help="JSON file with ICL examples; if not given, zero-shot without ICL.")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    icl_examples = []
    if args.icl and args.icl.exists():
        icl_examples = json.loads(args.icl.read_text(encoding="utf-8"))
        import random
        random.Random(args.shuffle_seed).shuffle(icl_examples)

    schema = SCHEMA_ZH if args.lang == "zh" else SCHEMA_EN
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_failed = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)
            messages = build_prompt(doc["text"], args.lang, icl_examples)
            try:
                if args.provider == "openai":
                    raw = call_openai(args.model, messages, schema)
                elif args.provider == "deepseek":
                    raw = call_deepseek(args.model, messages)
                else:
                    raw = call_vllm(args.endpoint, args.model, messages)
                parsed = json.loads(raw)
                preds = to_predictions_format(parsed)
            except Exception as e:
                n_failed += 1
                preds = []
                print(f"FAIL doc={doc['id']}: {e}")
            fout.write(json.dumps({"id": doc["id"], "predictions": preds,
                                   "raw_response": raw if 'raw' in dir() else ""},
                                  ensure_ascii=False) + "\n")
            time.sleep(0.05)
    print(f"Done. {n_failed} failures.")

if __name__ == "__main__":
    main()
