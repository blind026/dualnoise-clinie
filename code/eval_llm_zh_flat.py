"""
DualNoise-ClinIE — LLM zero-shot eval (ZH flat-NER + EN section-header KV)
============================================================================
Schema:
  - ZH: 6 CCKS entity types (disease_diagnosis, anatomical_site, ...)
  - EN: 9 MTSamples section-header keys (chief_complaint, history_of_present_illness, ...)

Provider support: openai (GPT-4o), deepseek, vllm (Qwen2.5-7B served locally).

Usage
-----
python eval_llm_zh_flat.py \
    --model gpt-4o-2024-08-06 --provider openai \
    --lang zh \
    --input data/processed/ccks_cells/L3_q1.0.jsonl \
    --output experiments/predictions/gpt4o_zh_le3.json \
    --max-docs 200
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Dict, List

ENTITY_TYPES_ZH = ["disease_diagnosis", "imaging_exam", "lab_exam",
                   "surgery", "medication", "anatomical_site"]

ENTITY_TYPES_EN = ["chief_complaint", "history_of_present_illness", "past_medical_history",
                   "social_family_history", "medications_allergies", "physical_examination",
                   "diagnostic_studies", "assessment_diagnosis", "plan_treatment"]

PROMPT_SYSTEM_ZH = """你是一名临床信息抽取专家，从扫描-OCR 转录的中文病历片段中按实体类型抽取关键信息。
严格遵循以下规则：
1. 只抽取输入文本中逐字出现的实体；禁止改写、释义、合成或归纳。
2. 不进行任何医学推断或诊断扩展；不要补充输入中未明确出现的内容。
3. 输入文本可能含有 OCR 噪声（错字、多余空格、视觉相似字符替换等），按文本字面抽取，不要纠正噪声。
4. 输出必须严格符合下方提供的 JSON Schema；任何字段若未在输入中出现，对应数组留空（[]）。
5. 同一字段下的多个实体按其在原文中出现的顺序排列。
6. 实体类型说明：
   - disease_diagnosis: 疾病和诊断（如"贲门低分化腺癌"、"骨髓抑制"）
   - imaging_exam: 影像检查（如"胸部CT"、"腹部彩超"）
   - lab_exam: 实验室检验（如"WBC"、"CA199"、"D-二聚体"）
   - surgery: 手术（如"根治性全胃切除术"）
   - medication: 药物（如"奥沙利铂"、"替吉奥"）
   - anatomical_site: 解剖部位（如"贲门小弯侧"、"右肺上叶"）"""

PROMPT_SYSTEM_EN = """You are a clinical information extraction expert. From OCR-transcribed English clinical reports, extract the FULL VERBATIM CONTENT of each section, organised by section type.

This is a SEMI-STRUCTURED extraction task. Each clinical report has section headers (e.g., "CHIEF COMPLAINT:", "PAST MEDICAL HISTORY:", "PHYSICAL EXAMINATION:"). For each section that appears in the input, extract the entire content of that section as a single string.

Follow these rules strictly:
1. The value for each section MUST be the verbatim content under that section header in the input — exactly as it appears, copying the full passage.
2. Do NOT paraphrase, summarise, abridge, or split into entities. Return the full section content as one continuous string.
3. The input may contain OCR noise (typos, extra whitespace, visually-similar character substitutions); copy the noisy text verbatim without correction.
4. Output must conform exactly to the provided JSON schema. Each section field is a string (or an empty string "" if that section is not present in the input).
5. Map document section headers to schema keys as follows:
   - chief_complaint: text under "CHIEF COMPLAINT"
   - history_of_present_illness: text under "HPI" / "HISTORY OF PRESENT ILLNESS" / "SUBJECTIVE" / "HISTORY"
   - past_medical_history: text under "PMH" / "PAST MEDICAL HISTORY" / "PAST SURGICAL HISTORY" / "MEDICAL HISTORY"
   - social_family_history: text under "FAMILY HISTORY" / "SOCIAL HISTORY"
   - medications_allergies: text under "MEDICATIONS" / "ALLERGIES" / "CURRENT MEDICATIONS"
   - physical_examination: text under "PHYSICAL EXAM" / "EXAMINATION" / "OBJECTIVE" / "VITAL SIGNS" / "REVIEW OF SYSTEMS" / "HEENT"
   - diagnostic_studies: text under "DIAGNOSTIC STUDIES" / "LABORATORY DATA" / "IMAGING" / "RADIOLOGY" / "EKG" / "CT SCAN" / "MRI"
   - assessment_diagnosis: text under "ASSESSMENT" / "DIAGNOSIS" / "IMPRESSION" / "PREOPERATIVE DIAGNOSIS" / "POSTOPERATIVE DIAGNOSIS"
   - plan_treatment: text under "PLAN" / "TREATMENT" / "PROCEDURE" / "OPERATION" / "HOSPITAL COURSE" / "ANESTHESIA"
6. If a section heading appears with no content, return "" for that key. If multiple chunks fall under the same key (e.g., two SUBJECTIVE-like blocks), concatenate them with a single newline."""

SCHEMA_ZH = {
    "type": "object", "additionalProperties": False,
    "required": ENTITY_TYPES_ZH,
    "properties": {k: {"type": "array", "items": {"type": "string"}} for k in ENTITY_TYPES_ZH},
}

SCHEMA_EN = {
    "type": "object", "additionalProperties": False,
    "required": ENTITY_TYPES_EN,
    # Section-level: value is the full section content as a string (or "" if absent)
    "properties": {k: {"type": "string"} for k in ENTITY_TYPES_EN},
}


def build_prompt(report_text: str, lang: str = "zh") -> List[Dict]:
    if lang == "en":
        return [
            {"role": "system", "content": PROMPT_SYSTEM_EN},
            {"role": "system", "content": f"OUTPUT SCHEMA:\n{json.dumps(SCHEMA_EN, ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": f"REPORT TEXT:\n{report_text}"},
        ]
    return [
        {"role": "system", "content": PROMPT_SYSTEM_ZH},
        {"role": "system", "content": f"OUTPUT SCHEMA:\n{json.dumps(SCHEMA_ZH, ensure_ascii=False, indent=2)}"},
        {"role": "user", "content": f"病历文本:\n{report_text}"},
    ]


def to_predictions(text: str, parsed: Dict, lang: str = "zh") -> List[Dict]:
    """Convert parsed JSON output to [{key, value, span}] list.

    ZH: parsed is {entity_type: [val, ...]} — multiple short values per type.
    EN: parsed is {section_key: section_content_str} — one long string per section.
    """
    if lang == "en":
        out = []
        if not isinstance(parsed, dict):
            return out
        for k in ENTITY_TYPES_EN:
            v = parsed.get(k)
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            idx = text.find(v)
            if idx >= 0:
                out.append({"key": k, "value": v, "span": [idx, idx + len(v)]})
            else:
                out.append({"key": k, "value": v, "span": [-1, -1]})
        return out
    # ZH path
    valid_types = ENTITY_TYPES_ZH
    out = []
    if not isinstance(parsed, dict):
        return out
    for k, vs in parsed.items():
        if k not in valid_types or not isinstance(vs, list):
            continue
        cursor = 0
        for v in vs:
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            idx = text.find(v, cursor)
            if idx < 0:
                idx = text.find(v)
            if idx >= 0:
                out.append({"key": k, "value": v, "span": [idx, idx + len(v)]})
                cursor = idx + len(v)
            else:
                out.append({"key": k, "value": v, "span": [-1, -1]})
    return out


def call_openai(model, messages, schema):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=2048,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "extraction", "schema": schema, "strict": False}},
    )
    return resp.choices[0].message.content


def call_deepseek(model, messages):
    from openai import OpenAI
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key=os.environ["DEEPSEEK_API_KEY"])
    # Disable thinking mode (default is enabled). Non-thinking is much faster
    # and less prone to API rate-limit timeouts.
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=2048,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    return resp.choices[0].message.content


def call_vllm(endpoint, model, messages):
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
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--max-docs", type=int, default=200)
    args = ap.parse_args()
    schema = SCHEMA_EN if args.lang == "en" else SCHEMA_ZH

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0; n_failed = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            doc = json.loads(line)
            if n_done >= args.max_docs: break
            messages = build_prompt(doc["text"], args.lang)
            raw = ""
            try:
                if args.provider == "openai":
                    raw = call_openai(args.model, messages, schema)
                elif args.provider == "deepseek":
                    raw = call_deepseek(args.model, messages)
                else:
                    raw = call_vllm(args.endpoint, args.model, messages)
                parsed = json.loads(raw)
                preds = to_predictions(doc["text"], parsed, args.lang)
            except Exception as e:
                n_failed += 1
                preds = []
                print(f"FAIL {doc['id']}: {e}", flush=True)
            fout.write(json.dumps({"id": doc["id"], "predictions": preds,
                                   "raw_response": raw}, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ... {n_done} done, {n_failed} failed", flush=True)
            time.sleep(0.03)
    print(f"Done. {n_done} processed, {n_failed} failures. -> {args.output}")


if __name__ == "__main__":
    main()
