"""
Subtask 3: Answer Generation
- Generate a text answer (≤75 words) to the patient's question using only the clinical note.
- Professional register, grounded in the note, no citations in the answer text.
- Few-shot: dev examples (question + note → reference answer).
- Ensemble: o3 + gpt-5.2 + gpt-5.1 (or configurable), then select best or merge.
"""

import os
import re
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Always load .env from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)

# Azure OpenAI SDK
AZURE_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/") + ("/" if os.getenv("AZURE_OPENAI_ENDPOINT") else "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_GPT52_API_KEY") or os.getenv("AZURE_O3_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or os.getenv("AZURE_DEPLOYMENT_BEST_1", "gpt-5.2")
_ensemble = (os.getenv("AZURE_ENSEMBLE_DEPLOYMENTS") or "o3,gpt-5.2,gpt-5.1").strip()
ENSEMBLE_DEPLOYMENTS = [d.strip() for d in _ensemble.split(",") if d.strip()]

# Few-shot: number of dev examples (10 = our best 33.53 on dev; 19 = LAMAR)
TASK3_FEW_SHOT_N = int(os.getenv("TASK3_FEW_SHOT_N", "10"))
# 80/20 dev split: hold-out case IDs (e.g. "4,9,14,19") excluded from few-shot/similar-case pool
_HOLDOUT_IDS_RAW = (os.getenv("TASK3_HOLDOUT_IDS") or "").strip()
TASK3_HOLDOUT_IDS = set(x.strip() for x in _HOLDOUT_IDS_RAW.split(",") if x.strip())
O3_FEW_SHOT_N = int(os.getenv("TASK3_O3_FEW_SHOT_N", "3"))  # shorter prompt for o3
# Max words in answer (Task 3 requirement: ≤75 words)
MAX_ANSWER_WORDS = 75
# Evidence-first: use Task 2 predictions to show only evidence sentences in the note (1=on)
TASK3_USE_TASK2_EVIDENCE = os.getenv("TASK3_USE_TASK2_EVIDENCE", "1").strip() in ("1", "true", "yes")
# Path to Task 2 submission JSON (case_id -> prediction: list of sentence IDs)
_TASK2_SUB = _REPO_ROOT / "task2" / "submission_Subtask2" / "submission.json"
TASK2_SUBMISSION_PATH = Path(os.getenv("TASK2_SUBMISSION", str(_TASK2_SUB)))
# Reformulation: second LLM pass to rewrite answer to ≤75 words, professional (LAMAR-style)
TASK3_REFORMULATION = os.getenv("TASK3_REFORMULATION", "1").strip() in ("1", "true", "yes")
TASK3_REFORMULATION_DEPLOYMENT = os.getenv("TASK3_REFORMULATION_DEPLOYMENT", "gpt-5.2")
# ICML-style: strict format + good/bad example in prompt
TASK3_ICML_STYLE = os.getenv("TASK3_ICML_STYLE", "1").strip() in ("1", "true", "yes")
# LLM-generated exemplars (LAMAR-style): use our model's answers on dev as few-shot instead of gold
TASK3_LLM_EXEMPLARS_PATH = os.getenv("TASK3_LLM_EXEMPLARS_PATH", "").strip()
# Reformulation style: "default" = full rewrite; "soft" = minimal wording change, preserve overlap (BLEU/ROUGE)
TASK3_REFORMULATION_STYLE = (os.getenv("TASK3_REFORMULATION_STYLE", "default") or "default").strip().lower()
# Citation-then-strip: ask model to cite note sentences as [1], [2]; then strip those for final output (encourages grounding)
TASK3_CITATION_THEN_STRIP = os.getenv("TASK3_CITATION_THEN_STRIP", "0").strip() in ("1", "true", "yes")
# Evidence-anchored: instruct model to construct answer by rephrasing/reshuffling evidence sentences only (LAMAR-inspired)
TASK3_EVIDENCE_ANCHORED = os.getenv("TASK3_EVIDENCE_ANCHORED", "0").strip() in ("1", "true", "yes")
# V2 faithful pipeline: two-stage (cite then rephrase from cited sentences only), gold few-shot with evidence mapping
TASK3_FAITHFUL = os.getenv("TASK3_FAITHFUL", "0").strip() in ("1", "true", "yes")
# Skip Stage 2: use Stage 1 draft with citations stripped (no reformulation call)
TASK3_SKIP_STAGE2 = os.getenv("TASK3_SKIP_STAGE2", "0").strip().lower() in ("1", "true", "yes")
# Stage 2 minimal edit: only strip citations + trim to 75 words (no LLM call) — preserves BLEU/ROUGE
TASK3_STAGE2_MINIMAL_EDIT = os.getenv("TASK3_STAGE2_MINIMAL_EDIT", "0").strip().lower() in ("1", "true", "yes")
# Similar-case few-shot: rank dev examples by similarity to current case, use top N (better overlap)
TASK3_SIMILAR_FEW_SHOT = os.getenv("TASK3_SIMILAR_FEW_SHOT", "0").strip().lower() in ("1", "true", "yes")
# Ensemble pick-best: score each model's answer vs note with BERTScore, use highest (when ensemble has 2+ answers)
TASK3_ENSEMBLE_PICK_BEST = os.getenv("TASK3_ENSEMBLE_PICK_BEST", "0").strip().lower() in ("1", "true", "yes")
# Score-maximizer mode: single-stage, explanatory style, multi-candidate, gpt-5.2-chat
TASK3_SCORE_MAX = os.getenv("TASK3_SCORE_MAX", "0").strip().lower() in ("1", "true", "yes")
TASK3_SCORE_MAX_MODEL = os.getenv("TASK3_SCORE_MAX_MODEL", "gpt-5.2-chat").strip()
TASK3_SCORE_MAX_CANDIDATES = int(os.getenv("TASK3_SCORE_MAX_CANDIDATES", "3"))
TASK3_SCORE_MAX_TEMP = float(os.getenv("TASK3_SCORE_MAX_TEMP", "0.7"))
# Experiment tag for output folder
TASK3_EXP_TAG = os.getenv("TASK3_EXP_TAG", "").strip()
# Final-mode: optimize for reference-style similarity (not note similarity)
TASK3_REFERENCE_STYLE_MODE = os.getenv("TASK3_REFERENCE_STYLE_MODE", "0").strip().lower() in ("1", "true", "yes")
# Context expansion: evidence +/- window sentences, optionally plus full note
TASK3_CONTEXT_WINDOW = int(os.getenv("TASK3_CONTEXT_WINDOW", "2"))
TASK3_INCLUDE_FULL_NOTE = os.getenv("TASK3_INCLUDE_FULL_NOTE", "1").strip().lower() in ("1", "true", "yes")
# Candidate generation: multiple samples per non-o3 model
TASK3_SAMPLES_PER_MODEL = int(os.getenv("TASK3_SAMPLES_PER_MODEL", "3"))
_sample_temps_raw = (os.getenv("TASK3_SAMPLE_TEMPS") or "0.0,0.2,0.4").strip()
TASK3_SAMPLE_TEMPS = [float(x.strip()) for x in _sample_temps_raw.split(",") if x.strip()]
# Stage 2 rewrite aligned to dev-answer style
TASK3_REWRITE_DEPLOYMENT = os.getenv("TASK3_REWRITE_DEPLOYMENT", "gpt-5.2").strip()
TASK3_REWRITE_FEW_SHOT_N = int(os.getenv("TASK3_REWRITE_FEW_SHOT_N", "8"))
# Stage 3 rerank against similar dev gold answers
TASK3_REF_RERANK_TOPK = int(os.getenv("TASK3_REF_RERANK_TOPK", "3"))
# ===== NUCLEAR PIPELINE: targets ALL 6 leaderboard metrics =====
TASK3_NUCLEAR = os.getenv("TASK3_NUCLEAR", "0").strip().lower() in ("1", "true", "yes")
TASK3_NUCLEAR_CANDIDATES = int(os.getenv("TASK3_NUCLEAR_CANDIDATES", "3"))
_nuclear_temps_raw = (os.getenv("TASK3_NUCLEAR_TEMPS") or "0.0,0.4,0.8").strip()
TASK3_NUCLEAR_TEMPS = [float(x.strip()) for x in _nuclear_temps_raw.split(",") if x.strip()]
TASK3_NUCLEAR_RERANK_TOPK = int(os.getenv("TASK3_NUCLEAR_RERANK_TOPK", "5"))

# gpt-5.2-chat: optional dedicated endpoint + api-version + key
AZURE_GPT52_CHAT_ENDPOINT = (os.getenv("AZURE_GPT52_CHAT_ENDPOINT") or AZURE_ENDPOINT).rstrip("/") + "/"
AZURE_GPT52_CHAT_API_VERSION = os.getenv("AZURE_GPT52_CHAT_API_VERSION", "2025-04-01-preview")
AZURE_GPT52_CHAT_API_KEY = os.getenv("AZURE_GPT52_CHAT_API_KEY") or AZURE_API_KEY

try:
    from openai import AzureOpenAI
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    ) if AZURE_API_KEY else None
    o3_api_version = os.getenv("AZURE_OPENAI_O3_API_VERSION", "2025-01-01-preview")
    azure_client_o3 = AzureOpenAI(
        api_version=o3_api_version,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    ) if AZURE_API_KEY and AZURE_ENDPOINT else None
    azure_client_gpt52_chat = AzureOpenAI(
        api_version=AZURE_GPT52_CHAT_API_VERSION,
        azure_endpoint=AZURE_GPT52_CHAT_ENDPOINT,
        api_key=AZURE_GPT52_CHAT_API_KEY,
    ) if AZURE_GPT52_CHAT_API_KEY and AZURE_GPT52_CHAT_ENDPOINT else None
except Exception as e:
    azure_client = None
    azure_client_o3 = None
    azure_client_gpt52_chat = None
    print(f"Azure OpenAI init: {e}")


def parse_qa_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """Parse archehr-qa.xml into list of cases."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = []
    for case_el in root.findall(".//case"):
        case_id = case_el.get("id", "")
        patient_question_el = case_el.find("patient_question")
        patient_question = ""
        if patient_question_el is not None:
            phrase = patient_question_el.find("phrase")
            if phrase is not None and (phrase.text or "").strip():
                patient_question = (phrase.text or "").strip()
            else:
                patient_question = (patient_question_el.text or "").strip()
        
        def el_text(el: Optional[ET.Element]) -> str:
            return "".join(el.itertext()).strip() if el is not None else ""
        
        clinician_el = case_el.find("clinician_question")
        clinician_question = el_text(clinician_el)
        note_el = case_el.find("note_excerpt")
        note_excerpt = el_text(note_el)
        sentences_el = case_el.find("note_excerpt_sentences")
        sentences = []
        if sentences_el is not None:
            for s in sentences_el.findall("sentence"):
                sid = s.get("id", "")
                text = el_text(s)
                sentences.append({"id": sid, "text": text})
        cases.append({
            "case_id": case_id,
            "patient_question": patient_question,
            "clinician_question": clinician_question,
            "note_excerpt": note_excerpt,
            "sentences": sentences,
        })
    return cases


def load_key_reference_answers(key_path: Path) -> Dict[str, str]:
    """Load reference clinician answers (without citations) per case."""
    if not key_path.exists():
        return {}
    with open(key_path, "r", encoding="utf-8") as f:
        key_json = json.load(f)
    out = {}
    for case in key_json:
        case_id = case["case_id"]
        answer = case.get("clinician_answer_without_citations", "").strip()
        if answer:
            out[case_id] = answer
    return out


def load_task2_evidence_map(submission_path: Path) -> Dict[str, List[str]]:
    """Load Task 2 submission: case_id -> list of predicted evidence sentence IDs."""
    if not submission_path.exists():
        return {}
    with open(submission_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["case_id"]: list(item["prediction"]) for item in data if item.get("prediction")}


def load_llm_exemplars(json_path: Path) -> Dict[str, str]:
    """Load LLM-generated exemplar answers: case_id -> answer text (from generate_exemplars.py)."""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return {k: (v if isinstance(v, str) else v.get("prediction", "")) for k, v in data.items()}
    out = {}
    for item in data:
        cid = item.get("case_id", "")
        ans = item.get("prediction", item.get("answer", ""))
        if cid and isinstance(ans, str):
            out[cid] = ans
    return out


def load_gold_evidence_and_answers(key_path: Path) -> Dict[str, Dict]:
    """Load gold evidence IDs + answer per case for faithful few-shot. Returns {case_id: {evidence_ids, answer}}."""
    if not key_path.exists():
        return {}
    with open(key_path, "r", encoding="utf-8") as f:
        key_json = json.load(f)
    out = {}
    for case in key_json:
        cid = case["case_id"]
        # Evidence: essential + supplementary sentence IDs
        eids = []
        for a in case.get("answers", []):
            if a.get("relevance") in ("essential", "supplementary"):
                eids.append(a["sentence_id"])
        answer = case.get("clinician_answer_without_citations", "").strip()
        out[cid] = {"evidence_ids": sorted(eids, key=lambda x: int(x) if x.isdigit() else 0), "answer": answer}
    return out


def build_faithful_stage1_prompt(
    case: Dict[str, Any],
    evidence_ids: Optional[List[str]],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    gold_evidence_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """Stage 1: Generate answer from evidence sentences with citations. Few-shot shows evidence→answer mapping."""
    lines = [
        "You are a medical expert answering patient questions using only the relevant sentences from their clinical note.",
        "",
        "Instructions:",
        f"- Your answer must be between 70 and {MAX_ANSWER_WORDS} words.",
        "- Rephrase the relevant sentences to directly address the patient's question. Stay very close to the original wording.",
        "- Use the same medical terms, procedure names, test names, and values as in the note sentences.",
        "- Do NOT add interpretation, reasoning, or information not stated in the sentences.",
        "- Do NOT use phrases like 'According to the note' or 'Based on the record'.",
        "- Refer to the patient in the third person ('the patient') when describing what happened.",
        "- Cite each sentence you use with its ID in brackets, e.g. [2], [7].",
        "",
    ]
    # Few-shot: show evidence sentences → gold answer
    if few_shot_examples and gold_evidence_map:
        lines.append("Examples:")
        for ex in few_shot_examples:
            c = ex["case"]
            cid = c["case_id"]
            gold = gold_evidence_map.get(cid)
            if not gold:
                continue
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            lines.append("Relevant note sentences:")
            eid_set = set(gold["evidence_ids"])
            for s in c["sentences"]:
                if s["id"] in eid_set:
                    lines.append(f"  {s['id']}: {s['text']}")
            lines.append("Answer: " + gold["answer"])
            lines.append("")
        lines.append("Now generate an answer for the following case using the same style:")
        lines.append("")
    
    lines.extend([
        "Patient question:",
        case["patient_question"],
        "",
        "Clinician question:",
        case["clinician_question"],
        "",
        "Relevant note context:",
    ])
    lines.append(_expanded_context_text_for_case(case, evidence_ids))
    lines.extend([
        "",
        f"Rephrase these sentences into a {MAX_ANSWER_WORDS}-word answer addressing the patient's question. Cite sentence IDs as [2], [7]. Stay close to the original wording. Output only the answer.",
    ])
    return "\n".join(lines)


def build_faithful_stage2_prompt(
    patient_question: str,
    clinician_question: str,
    cited_sentences: List[Dict[str, str]],
    draft_answer: str,
) -> str:
    """Stage 2: Given ONLY the cited sentences, rephrase into final answer. No other content allowed."""
    lines = [
        "You are a medical expert. Rewrite the draft answer below using ONLY the provided note sentences.",
        "",
        "Rules:",
        f"- Output must be between 70 and {MAX_ANSWER_WORDS} words.",
        "- Rephrase and reshuffle the sentences to answer the question. Stay very close to the original sentence wording.",
        "- Do NOT add any content, interpretation, or reasoning not present in these sentences.",
        "- Do NOT include citation markers [1], [2] in the output.",
        "- Refer to the patient in the third person ('the patient').",
        "- Use the same medical terms and values from the sentences.",
        "",
        "Patient question: " + patient_question,
        "Clinician question: " + clinician_question,
        "",
        "Note sentences to use:",
    ]
    for s in cited_sentences:
        lines.append(f"  {s['id']}: {s['text']}")
    lines.extend([
        "",
        "Draft answer (with citations):",
        draft_answer,
        "",
        f"Rewrite into a clean {MAX_ANSWER_WORDS}-word answer using only the sentences above. No citations. Output only the answer.",
    ])
    return "\n".join(lines)


def build_score_max_prompt(
    case: Dict[str, Any],
    evidence_ids: Optional[List[str]],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    gold_evidence_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """Score-maximizer prompt: single-stage, explanatory style matching gold answers."""
    lines = [
        "You are a clinician explaining medical information to a patient based on their clinical note.",
        "",
        "Your task: Write a clear, explanatory answer to the patient's question.",
        "",
        "Style guidelines:",
        f"- Write EXACTLY 70 to {MAX_ANSWER_WORDS} words. Use all available space.",
        "- Explain cause-and-effect: WHY things were done, not just what happened.",
        "- Use medical terms from the note (procedure names, test names, medication names, values) but explain their purpose.",
        "- Write in third person ('the patient was given...', 'the procedure showed...').",
        "- Do NOT include dates or timestamps.",
        "- Do NOT start with 'Based on the note' or 'According to the records'.",
        "- Do NOT include citation markers like [1], [2].",
        "- End with a complete sentence.",
        "- Connect ideas with causal language: 'due to', 'which caused', 'resulting in', 'to allow', 'because of', 'leading to'.",
        "",
    ]
    # Few-shot: show gold examples to teach the style
    if few_shot_examples and gold_evidence_map:
        lines.append("Examples of good answers:")
        for ex in few_shot_examples:
            c = ex["case"]
            cid = c["case_id"]
            gold = gold_evidence_map.get(cid)
            if not gold or not gold.get("answer"):
                continue
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            # Show evidence sentences (not full note, to keep prompt focused)
            eid_set = set(gold["evidence_ids"])
            lines.append("Key note sentences:")
            for s in c["sentences"]:
                if s["id"] in eid_set:
                    lines.append(f"  {s['id']}: {s['text']}")
            lines.append(f"Answer ({len(gold['answer'].split())} words): " + gold["answer"])
            lines.append("")
    elif few_shot_examples:
        lines.append("Examples of good answers:")
        for ex in few_shot_examples:
            c = ex["case"]
            ref = ex["reference_answer"]
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            lines.append(f"Answer ({len(ref.split())} words): " + ref)
            lines.append("")

    lines.append("Now write an answer for this case:")
    lines.append("")
    lines.extend([
        "Patient question:",
        case["patient_question"],
        "",
        "Clinician question:",
        case["clinician_question"],
        "",
    ])
    # Show evidence sentences + full note context
    if evidence_ids and case.get("sentences"):
        eid_set = set(evidence_ids)
        evidence_sents = [s for s in case["sentences"] if s["id"] in eid_set]
        if evidence_sents:
            lines.append("Key note sentences (most relevant):")
            for s in evidence_sents:
                lines.append(f"  {s['id']}: {s['text']}")
            lines.append("")
    # Also provide full note for context
    lines.append("Full clinical note excerpt:")
    if case.get("sentences"):
        for s in case["sentences"]:
            lines.append(f"  {s['id']}: {s['text']}")
    elif case.get("note_excerpt"):
        lines.append(case["note_excerpt"])
    lines.extend([
        "",
        f"Write a {MAX_ANSWER_WORDS}-word explanatory answer. Explain causes and effects. Use medical terms from the note. No citations. No dates. End with a complete sentence.",
    ])
    return "\n".join(lines)


def load_few_shot_examples(
    key_path: Path,
    xml_path: Path,
    case_ids: List[str],
    cases_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    exemplar_answers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Build few-shot examples: list of {case, reference_answer}. Use exemplar_answers if provided (LLM-generated), else gold key."""
    answer_map = load_key_reference_answers(key_path) if key_path.exists() else {}
    if cases_by_id is None:
        if not xml_path.exists():
            return []
        cases_list = parse_qa_xml(xml_path)
        cases_by_id = {c["case_id"]: c for c in cases_list}
    # Prefer LLM exemplars when provided
    ref_map = exemplar_answers if exemplar_answers else answer_map
    if not ref_map:
        return []
    examples = []
    for cid in case_ids:
        if cid not in ref_map or cid not in cases_by_id:
            continue
        examples.append({
            "case": cases_by_id[cid],
            "reference_answer": ref_map[cid],
        })
    return examples


def truncate_to_max_words(text: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    """Truncate answer to max_words (Task 3 requirement: ≤75 words)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def strip_citations(text: str) -> str:
    """Remove inline citation markers [1], [2], [1,2], etc. Collapse spaces."""
    if not text:
        return text
    # Match [1], [2], [1, 2], [ 1 ], etc.
    out = re.sub(r"\[\s*\d+\s*(?:\s*,\s*\d+\s*)*\]", "", text)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _note_text_for_case(case: Dict[str, Any], evidence_ids: Optional[List[str]] = None) -> str:
    """Get note text for a case; if evidence_ids provided, show only those sentences (order preserved)."""
    if evidence_ids is not None and evidence_ids and case.get("sentences"):
        id_set = set(evidence_ids)
        parts = []
        for s in case["sentences"]:
            if s["id"] in id_set:
                parts.append(f"  {s['id']}: {s['text']}")
        if parts:
            return "\n".join(parts)
    # Full note
    if case.get("note_excerpt"):
        return case["note_excerpt"]
    if case.get("sentences"):
        return "\n".join(f"  {s['id']}: {s['text']}" for s in case["sentences"])
    return ""


def _expanded_context_text_for_case(case: Dict[str, Any], evidence_ids: Optional[List[str]] = None) -> str:
    """Build context for final pipeline:
    evidence sentences + surrounding window, optionally plus full note.
    """
    if not case.get("sentences"):
        return _note_text_for_case(case, None)
    sentences = case["sentences"]
    if not evidence_ids:
        return _note_text_for_case(case, None)
    id_to_idx = {s["id"]: i for i, s in enumerate(sentences)}
    selected_idx = set()
    for eid in evidence_ids:
        if eid not in id_to_idx:
            continue
        i = id_to_idx[eid]
        lo = max(0, i - max(0, TASK3_CONTEXT_WINDOW))
        hi = min(len(sentences) - 1, i + max(0, TASK3_CONTEXT_WINDOW))
        for j in range(lo, hi + 1):
            selected_idx.add(j)
    if not selected_idx:
        return _note_text_for_case(case, None)
    parts = ["Context sentences (evidence + surrounding):"]
    for j in sorted(selected_idx):
        s = sentences[j]
        parts.append(f"  {s['id']}: {s['text']}")
    if TASK3_INCLUDE_FULL_NOTE:
        parts.append("")
        parts.append("Full note:")
        for s in sentences:
            parts.append(f"  {s['id']}: {s['text']}")
    return "\n".join(parts)


def _text_for_similarity(case: Dict[str, Any]) -> str:
    """Concatenate question + note for similarity scoring (no dependency)."""
    parts = [
        (case.get("patient_question") or ""),
        (case.get("clinician_question") or ""),
        _note_text_for_case(case, None),
    ]
    return " ".join(parts).lower()


def _rank_few_shot_by_similarity(
    few_shot_list: List[Dict[str, Any]], current_case: Dict[str, Any], top_k: int
) -> List[Dict[str, Any]]:
    """Rank few-shot examples by word-overlap similarity to current case; return top_k."""
    if not few_shot_list or top_k <= 0:
        return few_shot_list[:top_k] if top_k > 0 else []
    current_text = _text_for_similarity(current_case)
    current_words = set(re.findall(r"\b\w+\b", current_text))
    if not current_words:
        return few_shot_list[:top_k]
    scored = []
    for ex in few_shot_list:
        other_text = _text_for_similarity(ex["case"])
        other_words = set(re.findall(r"\b\w+\b", other_text))
        overlap = len(current_words & other_words) / max(len(current_words), 1)
        scored.append((overlap, ex))
    scored.sort(key=lambda x: -x[0])
    return [ex for _, ex in scored[:top_k]]


def _pick_best_candidate_by_bertscore(candidates: List[str], case: Dict[str, Any],
                                       evidence_ids: Optional[List[str]] = None) -> str:
    """Pick the candidate with highest BERTScore vs the note (proxy for reference similarity)."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    import sys
    eval_dir = _REPO_ROOT / "task2" / "Evaluation_Task2" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import torch
    from scorers.bert_scorer import BertScorer
    note_text = _note_text_for_case(case, None)
    if not note_text.strip():
        return candidates[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BertScorer(device=device)
    refs = [note_text] * len(candidates)
    scores = scorer.compute_scores(refs, candidates)
    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    return candidates[best_idx]


def truncate_at_sentence_boundary(text: str, max_words: int = MAX_ANSWER_WORDS) -> str:
    """Truncate at the last sentence boundary within max_words. Never cut mid-sentence."""
    if not text:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    # Try to find last sentence-ending punctuation within max_words
    truncated = " ".join(words[:max_words])
    # Find last sentence boundary (. ? !)
    best_end = -1
    for i, ch in enumerate(truncated):
        if ch in '.?!' and i > 0:
            # Check it's not part of an abbreviation (e.g., "Dr.", "e.g.")
            best_end = i
    if best_end > len(truncated) * 0.5:  # Only use if we keep at least 50% of the text
        return truncated[:best_end + 1].strip()
    # Fallback: simple word truncation
    return truncated.strip()


def build_nuclear_prompt(
    case: Dict[str, Any],
    evidence_ids: Optional[List[str]],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    gold_evidence_map: Optional[Dict[str, Dict]] = None,
) -> str:
    """Nuclear prompt: frame task as 'predict the clinician reference answer'.
    Optimized for all 6 evaluation metrics: BLEU, ROUGE, SARI, BERTScore, AlignScore, MEDCON.
    Key insight: AlignScore penalizes facts NOT in the reference, so we must be selective.
    """
    lines = [
        "You are writing a clinician's answer for a clinical QA evaluation.",
        "Your answer will be scored against a reference answer written by a clinical expert.",
        "The evaluation measures how closely your answer matches the reference in wording, meaning, and medical content.",
        "",
        "CRITICAL RULES:",
        "- Answer ONLY the patient's question. Do NOT include background, admission history, or tangential details.",
        "- Include only the medical facts most directly relevant to answering the question.",
        "- Every sentence must contribute to answering the question. Remove anything that does not.",
        f"- Write exactly 65 to {MAX_ANSWER_WORDS} words. End with a complete sentence.",
        "- Use clinical terminology from the note: drug names, procedure names, test results, values.",
        "- Write in third person ('the patient', 'he/she was').",
        "- Do NOT say 'according to the note', 'based on the records', etc.",
        "- Cite note sentence IDs as [1], [2] after relevant facts. These will be removed later.",
        "- Match the writing style of the example answers below as closely as possible.",
        "",
    ]
    # Few-shot: show gold examples as "reference answers"
    if few_shot_examples and gold_evidence_map:
        lines.append("EXAMPLE REFERENCE ANSWERS (study the style, structure, and level of detail):")
        for ex in few_shot_examples:
            c = ex["case"]
            cid = c["case_id"]
            gold = gold_evidence_map.get(cid)
            if not gold or not gold.get("answer"):
                continue
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            # Show evidence sentences
            eid_set = set(gold["evidence_ids"])
            ev_sents = [s for s in c["sentences"] if s["id"] in eid_set]
            if ev_sents:
                lines.append("Evidence sentences:")
                for s in ev_sents:
                    lines.append(f"  [{s['id']}] {s['text']}")
            answer_wc = len(gold["answer"].split())
            lines.append(f"Reference answer ({answer_wc}w): " + gold["answer"])
            lines.append("")
    elif few_shot_examples:
        lines.append("EXAMPLE REFERENCE ANSWERS (study the style, structure, and level of detail):")
        for ex in few_shot_examples:
            c = ex["case"]
            ref = ex["reference_answer"]
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            ref_wc = len(ref.split())
            lines.append(f"Reference answer ({ref_wc}w): " + ref)
            lines.append("")

    lines.append("=" * 40)
    lines.append("NOW WRITE YOUR ANSWER FOR THIS CASE:")
    lines.append("")
    lines.extend([
        "Patient question: " + case["patient_question"],
        "",
        "Clinician question: " + case["clinician_question"],
        "",
    ])
    # Show evidence sentences prominently, then full note as context
    if evidence_ids and case.get("sentences"):
        eid_set = set(evidence_ids)
        evidence_sents = [s for s in case["sentences"] if s["id"] in eid_set]
        if evidence_sents:
            lines.append("Most relevant note sentences:")
            for s in evidence_sents:
                lines.append(f"  [{s['id']}] {s['text']}")
            lines.append("")
    lines.append("Full clinical note:")
    if case.get("sentences"):
        for s in case["sentences"]:
            lines.append(f"  [{s['id']}] {s['text']}")
    elif case.get("note_excerpt"):
        lines.append(case["note_excerpt"])
    lines.extend([
        "",
        f"Write a 65-{MAX_ANSWER_WORDS} word answer that a clinical expert would write.",
        "Answer ONLY the question. Include only directly relevant medical facts. Match the example style. Cite sentences as [1], [2].",
        "Answer:",
    ])
    return "\n".join(lines)


def _pick_best_nuclear(
    candidates: List[str],
    current_case: Dict[str, Any],
    few_shot_pool: List[Dict[str, Any]],
    top_k: int = 5,
) -> str:
    """Nuclear reranking: composite BERTScore + ROUGE vs similar dev gold answers.
    Also penalizes incomplete sentences and wrong word counts.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Get similar dev gold answers as references
    pool = [ex for ex in few_shot_pool if ex.get("reference_answer", "").strip()]
    if not pool:
        return candidates[0]

    similar = _rank_few_shot_by_similarity(pool, current_case, max(1, top_k))
    refs = [ex["reference_answer"] for ex in similar if ex.get("reference_answer")]
    if not refs:
        return candidates[0]

    import sys
    eval_dir = _REPO_ROOT / "task2" / "Evaluation_Task2" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import torch
    from scorers.bert_scorer import BertScorer
    from scorers.rouge_scorer import RougeScorer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scorer = BertScorer(device=device)
    rouge_scorer = RougeScorer()

    composite_scores: List[float] = []
    for cand in candidates:
        # BERTScore vs each dev gold ref
        bert_preds = [cand] * len(refs)
        bert_scores = bert_scorer.compute_scores(refs, bert_preds)
        avg_bert = float(sum(bert_scores) / max(len(bert_scores), 1))

        # ROUGE-Lsum vs each dev gold ref
        rouge_vals = []
        for ref in refs:
            r = rouge_scorer.compute_overall_score([ref], [cand])
            rouge_vals.append(r.get("rougeLsum", 0.0))
        avg_rouge = float(sum(rouge_vals) / max(len(rouge_vals), 1))

        # Penalties
        wc = len(cand.split())
        # Length penalty: prefer 60-75 words
        if 60 <= wc <= MAX_ANSWER_WORDS:
            len_bonus = 0.02
        elif wc < 50:
            len_bonus = -0.05
        else:
            len_bonus = 0.0
        # Sentence completion bonus
        ends_well = 0.02 if cand.rstrip() and cand.rstrip()[-1] in '.?!' else -0.03

        # Composite: BERTScore (0.5) + ROUGE (0.3) + penalties (0.2)
        score = 0.5 * avg_bert + 0.3 * avg_rouge + len_bonus + ends_well
        composite_scores.append(score)

    best_idx = int(max(range(len(candidates)), key=lambda i: composite_scores[i]))
    return candidates[best_idx]


def build_reference_style_rewrite_prompt(
    case: Dict[str, Any],
    grounded_draft: str,
    note_context: str,
    style_examples: List[Dict[str, Any]],
) -> str:
    """Stage 2: aggressively align style to dev gold answers while staying grounded."""
    lines = [
        "You are writing clinician answers for a clinical QA dataset.",
        "",
        "Match the style, structure, and wording patterns of the example answers.",
        "",
        "Rules:",
        "- Use ONLY facts from the provided grounded draft and note.",
        "- Do NOT add outside medical knowledge.",
        "- Keep clinical terminology.",
        "- Follow the narrative structure of the examples.",
        "- 60-75 words.",
        "- Professional clinician tone.",
        "- Do not mention 'the note states'.",
        "- Produce a single paragraph answer.",
        "",
        "Style examples (dev gold answers):",
    ]
    for ex in style_examples:
        c = ex["case"]
        ref = ex["reference_answer"]
        lines.append("---")
        lines.append("Patient question: " + c.get("patient_question", ""))
        lines.append("Clinician question: " + c.get("clinician_question", ""))
        lines.append("Gold answer: " + ref)
        lines.append("")
    lines.extend([
        "Current case:",
        "Patient question: " + case["patient_question"],
        "Clinician question: " + case["clinician_question"],
        "",
        "Grounded draft:",
        grounded_draft,
        "",
        "Clinical note context:",
        note_context,
        "",
        "Output only the final answer paragraph:",
    ])
    return "\n".join(lines)


def _pick_best_candidate_by_dev_refs(
    candidates: List[str],
    current_case: Dict[str, Any],
    few_shot_pool: List[Dict[str, Any]],
) -> str:
    """Rerank candidates by similarity to top-k similar dev gold answers."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    pool = [ex for ex in few_shot_pool if ex.get("reference_answer", "").strip()]
    if not pool:
        return candidates[0]
    top_k = max(1, TASK3_REF_RERANK_TOPK)
    similar = _rank_few_shot_by_similarity(pool, current_case, top_k)
    refs = [ex["reference_answer"] for ex in similar if ex.get("reference_answer")]
    if not refs:
        refs = [ex["reference_answer"] for ex in pool[:top_k] if ex.get("reference_answer")]
    if not refs:
        return candidates[0]
    import sys
    eval_dir = _REPO_ROOT / "task2" / "Evaluation_Task2" / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    import torch
    from scorers.bert_scorer import BertScorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BertScorer(device=device)
    mean_scores: List[float] = []
    for cand in candidates:
        preds = [cand] * len(refs)
        scores = scorer.compute_scores(refs, preds)
        mean_scores.append(float(sum(scores) / max(len(scores), 1)))
    best_idx = int(max(range(len(candidates)), key=lambda i: mean_scores[i]))
    return candidates[best_idx]


def build_answer_prompt(
    case: Dict[str, Any],
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    evidence_ids: Optional[List[str]] = None,
) -> str:
    """Build prompt for answer generation with optional few-shot and evidence-filtered note."""
    if TASK3_EVIDENCE_ANCHORED:
        lines = [
            "You are a medical expert providing clear, accurate answers to patient questions based on relevant sentences from the clinical note.",
            "",
            "Instructions:",
            f"- Construct your answer by rephrasing and reorganizing the relevant note sentences to directly address the patient's question.",
            f"- Your answer MUST be between 65 and {MAX_ANSWER_WORDS} words. Use all the space available — aim for {MAX_ANSWER_WORDS} words.",
            "- Use the clinical terminology and phrasing from the note. Do not add medical reasoning, explanations, or information beyond what the note states.",
            "- Include specific clinical details from the note: test names, values, medication names, dates, procedures.",
            "- Do not use phrases like 'According to the note' or 'The note states'. Write as if you are the clinician directly informing the patient.",
            "- Cite supporting sentence IDs inline as [1], [2], etc. We will remove these citations afterward.",
            "",
        ]
    elif TASK3_CITATION_THEN_STRIP:
        lines = [
            "You are a clinical assistant. Generate a professional answer to the patient's question using only information from the clinical note excerpt.",
            "Cite the note: after a claim, add the sentence ID in brackets, e.g. [1] or [2] or [1,3] for multiple. Use the sentence numbers from the note (e.g. '1:', '2:').",
            "",
            "Requirements:",
            f"- Answer must be at most {MAX_ANSWER_WORDS} words (including citations).",
            "- Use professional/clinical register. Base your answer only on the provided note.",
            "- Cite which note sentences support each claim using [1], [2], etc. We will remove these citations for the final answer.",
            "- If the note does not fully answer the question, provide a faithful response based on available evidence.",
            "",
        ]
    else:
        lines = [
            "You are a clinical assistant. Generate a professional answer to the patient's question using only information from the clinical note excerpt.",
            "",
            "Requirements:",
            f"- Answer must be at most {MAX_ANSWER_WORDS} words.",
            "- Write in third person ('the patient', not 'you'). Use clear, explanatory sentences that directly address the question, matching the style of the example answers (not terse clinical note style).",
            "- Base your answer only on the provided clinical note. Do not add information not in the note.",
            "- Do not include citations or references in the answer text.",
            "- If the note does not fully answer the question, provide a faithful response based on available evidence.",
            "",
        ]
    if TASK3_ICML_STYLE and not TASK3_CITATION_THEN_STRIP and not TASK3_EVIDENCE_ANCHORED:
        lines.extend([
            "Output format: Output only the answer text. No labels, no prefixes, no citations like [1].",
            "Good example: \"The scan showed no new bleeding. Your symptoms are consistent with improvement.\"",
            "Bad example: \"According to the note [1] the scan showed...\" (has citations) or answers over 75 words.",
            "",
        ])
    
    if few_shot_examples:
        lines.append("Examples:")
        for ex in few_shot_examples:
            c, ref = ex["case"], ex["reference_answer"]
            lines.append("---")
            lines.append("Patient question: " + c["patient_question"])
            lines.append("Clinician question: " + c["clinician_question"])
            lines.append("Clinical note:")
            note = _note_text_for_case(c, None)
            if len(note) > 600:
                note = note[:600] + "..."
            lines.append(note)
            lines.append("Answer: " + ref)
            lines.append("")
        lines.append("Now generate an answer for the following case:")
        lines.append("")
    
    lines.extend([
        "Patient question:",
        case["patient_question"],
        "",
        "Clinician-interpreted question:",
        case["clinician_question"],
        "",
        "Clinical note excerpt:",
    ])
    note_text = _note_text_for_case(case, evidence_ids)
    lines.append(note_text if note_text else _note_text_for_case(case, None))
    if TASK3_EVIDENCE_ANCHORED:
        lines.extend([
            "",
            f"Construct a {MAX_ANSWER_WORDS}-word answer by rephrasing the relevant note sentences above. Include specific clinical details (values, names, dates). Cite sentence IDs as [1], [2], etc. Output only the answer.",
        ])
    elif TASK3_CITATION_THEN_STRIP:
        lines.extend([
            "",
            f"Generate a professional answer (at most {MAX_ANSWER_WORDS} words) using only the note above. Cite supporting sentence IDs as [1], [2], etc. Output only the answer with citations.",
        ])
    else:
        lines.extend([
            "",
            f"Generate a professional answer (at most {MAX_ANSWER_WORDS} words) using only information from the note above. No citations. Output only the answer text.",
        ])
    return "\n".join(lines)


def call_azure_chat(prompt: str, max_tokens: int = 500, temperature: float = 0.0, deployment: Optional[str] = None) -> str:
    """Call Azure OpenAI Chat Completions. o3 and gpt-5.2-chat get dedicated clients; retry on 429 and empty."""
    model = deployment or AZURE_DEPLOYMENT
    is_o3 = model.lower() in ("o3", "o3-pro")
    is_gpt52_chat = model.lower() == "gpt-5.2-chat"
    if is_gpt52_chat and azure_client_gpt52_chat:
        client = azure_client_gpt52_chat
    elif is_o3 and azure_client_o3:
        client = azure_client_o3
    else:
        client = azure_client
    if not client:
        return ""
    # gpt-5.2-chat only supports default temperature (1); do not pass temperature
    use_temp = not is_o3 and not is_gpt52_chat
    # gpt-5.2-chat is a reasoning model: needs high token cap for internal thinking (~128-192 reasoning tokens)
    cap = 2048 if is_o3 else (4096 if is_gpt52_chat else max_tokens)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if use_temp:
        kwargs["temperature"] = temperature
    kwargs["max_completion_tokens"] = cap
    max_retries = 4
    empty_retries = 2
    for attempt in range(max_retries):
        try:
            try:
                response = client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("max_completion_tokens", None)
                if not is_o3 and not is_gpt52_chat:
                    kwargs["max_tokens"] = max_tokens
                else:
                    kwargs["max_completion_tokens"] = cap
                response = client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None
            raw = choice.message.content if choice and choice.message else None
            # Reasoning models can return content as list of parts (e.g. [{"type":"text","text":"..."}])
            if isinstance(raw, list):
                content = " ".join(
                    p.get("text", p) if isinstance(p, dict) else str(p)
                    for p in raw
                    if (p.get("text") if isinstance(p, dict) else p)
                ).strip()
            else:
                content = (raw or "").strip() if isinstance(raw, str) else ""
            if content:
                return content
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            print(f"  Empty from {model}: finish_reason={finish_reason!r}")
            if is_gpt52_chat and choice and choice.message:
                raw = getattr(choice.message, "content", None)
                print(f"  DEBUG gpt-5.2-chat content type={type(raw).__name__!r} repr={repr(raw)[:300]!r}")
            if empty_retries > 0:
                empty_retries -= 1
                wait = 5
                print(f"  Waiting {wait}s then retry ({2 - empty_retries}/2)")
                time.sleep(wait)
                continue
            return ""
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "ratelimit" in err_str) and attempt < max_retries - 1:
                wait = 10
                print(f"  Rate limit (429), waiting {wait}s then retry ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"  API error ({model}): {e}")
            return ""
    return ""


def _reformulate_answer(draft: str, patient_question: str, deployment: str) -> str:
    """Second pass: rewrite to professional, ≤75 words, no citations (LAMAR-style). Style=soft preserves wording for BLEU/ROUGE."""
    if not draft or not deployment:
        return draft
    if TASK3_EVIDENCE_ANCHORED:
        # Evidence-anchored reformulation: strip citations, preserve note phrasing, fill to 75 words
        prompt = (
            "You are a clinical editor. Clean up the following draft answer:\n"
            "1. Remove all citation markers like [1], [2], etc.\n"
            "2. Keep the clinical terminology and specific details (values, names, procedures) exactly as written.\n"
            "3. The answer MUST be between 65 and " + str(MAX_ANSWER_WORDS) + " words. If it is shorter, expand with relevant detail from the draft. If longer, trim less important parts.\n"
            "4. Do not add any information not already in the draft.\n"
            "5. Output only the final answer text, nothing else.\n\n"
            f"Patient question: {patient_question}\n\n"
            f"Draft:\n{draft}\n\n"
            "Final answer:"
        )
    elif TASK3_REFORMULATION_STYLE == "soft":
        prompt = (
            "Shorten the following answer to at most " + str(MAX_ANSWER_WORDS) + " words and remove any citations or references like [1], [2]. "
            "Change wording as little as possible. Output only the shortened answer, nothing else.\n\n"
            f"Patient question: {patient_question}\n\n"
            f"Draft:\n{draft}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            "You are a clinical editor. Rewrite the following draft answer to be professional, "
            f"at most {MAX_ANSWER_WORDS} words, and with no citations or references. "
            "Preserve all factual content. Output only the rewritten answer, nothing else.\n\n"
            f"Patient question: {patient_question}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Rewritten answer:"
        )
    out = call_azure_chat(prompt, max_tokens=250, deployment=deployment)
    return (out or draft).strip()


def run_answer_pipeline(
    xml_path: Path,
    out_path: Path,
    key_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Run answer generation pipeline: few-shot + ensemble (or single model)."""
    if data_dir is None:
        data_dir = xml_path.parent.parent if xml_path.parent.name != "v1.4" else xml_path.parent
    
    cases = parse_qa_xml(xml_path)
    if limit:
        cases = cases[:limit]
    cases_by_id = {c["case_id"]: c for c in cases}
    
    # Evidence-first: load Task 2 predictions to filter note to evidence sentences
    evidence_map: Dict[str, List[str]] = {}
    if TASK3_USE_TASK2_EVIDENCE and TASK2_SUBMISSION_PATH.exists():
        evidence_map = load_task2_evidence_map(TASK2_SUBMISSION_PATH)
        print(f"Evidence-first: loaded Task 2 predictions for {len(evidence_map)} cases")
    elif TASK3_USE_TASK2_EVIDENCE:
        print("Evidence-first: Task 2 submission not found, using full note")
    
    # Few-shot: dev examples (gold or LLM-generated exemplars, LAMAR-style)
    few_shot_examples_all: List[Dict[str, Any]] = []
    exemplar_answers: Optional[Dict[str, str]] = None
    if TASK3_LLM_EXEMPLARS_PATH:
        exemplar_path = Path(TASK3_LLM_EXEMPLARS_PATH)
        if exemplar_path.exists():
            exemplar_answers = load_llm_exemplars(exemplar_path)
            print(f"LLM exemplars: loaded {len(exemplar_answers)} from {exemplar_path.name}")
    dev_key = key_path or (data_dir / "dev" / "archehr-qa_key.json")
    dev_xml = data_dir / "dev" / "archehr-qa.xml"
    if TASK3_FEW_SHOT_N > 0 and dev_xml.exists():
        # When similar-case ranking or nuclear: load all 20 dev so we can rank and take top N
        max_dev = 20 if (TASK3_SIMILAR_FEW_SHOT or TASK3_NUCLEAR) else min(TASK3_FEW_SHOT_N, 20)
        if TASK3_HOLDOUT_IDS:
            # 80/20 split: use only tune (non-hold-out) cases for few-shot/similar-case pool
            few_shot_case_ids = [str(i) for i in range(1, 21) if str(i) not in TASK3_HOLDOUT_IDS]
            print(f"Hold-out (excluded from few-shot): {sorted(TASK3_HOLDOUT_IDS)} -> {len(few_shot_case_ids)} tune examples")
        else:
            few_shot_case_ids = [str(i) for i in range(1, max_dev + 1)]
        few_shot_examples_all = load_few_shot_examples(
            dev_key, dev_xml, few_shot_case_ids, cases_by_id=None, exemplar_answers=exemplar_answers
        )
        print(f"Few-shot (dev): {len(few_shot_examples_all)} examples" + (" [LLM-generated]" if exemplar_answers else " [gold]") + (" [similar-case rank]" if TASK3_SIMILAR_FEW_SHOT else ""))
    
    use_ensemble = len(ENSEMBLE_DEPLOYMENTS) >= 1
    if use_ensemble:
        print(f"Ensemble: {ENSEMBLE_DEPLOYMENTS}")
    if TASK3_REFORMULATION:
        print(f"Reformulation: on (deployment={TASK3_REFORMULATION_DEPLOYMENT})")
    if TASK3_CITATION_THEN_STRIP:
        print("Citation-then-strip: on (generate with [1],[2] then strip)")
    
    if TASK3_EVIDENCE_ANCHORED:
        print("Evidence-anchored: on (rephrase note sentences, cite then strip)")
    
    # Faithful mode: load gold evidence→answer mapping for few-shot
    gold_evidence_map: Optional[Dict[str, Dict]] = None
    if TASK3_FAITHFUL or TASK3_SCORE_MAX or TASK3_NUCLEAR:
        gold_evidence_map = load_gold_evidence_and_answers(dev_key)
    if TASK3_FAITHFUL:
        stage_desc = "skip Stage 2 (draft only)" if TASK3_SKIP_STAGE2 else ("Stage 2 minimal-edit (strip+trim)" if TASK3_STAGE2_MINIMAL_EDIT else "two-stage cite+rephrase")
        print(f"Faithful mode: on ({stage_desc}, gold evidence->answer few-shot, {len(gold_evidence_map)} gold cases)")
    if TASK3_SIMILAR_FEW_SHOT:
        print(f"Similar-case few-shot: on (rank dev by similarity, top {TASK3_FEW_SHOT_N})")
    if TASK3_ENSEMBLE_PICK_BEST:
        print("Ensemble pick-best: on")
    if TASK3_REFERENCE_STYLE_MODE:
        print(
            "Reference-style mode: on (rewrite to dev style, rerank vs similar dev gold answers)"
        )
    if TASK3_NUCLEAR:
        print(f"NUCLEAR mode: on ({TASK3_NUCLEAR_CANDIDATES} candidates/model, temps={TASK3_NUCLEAR_TEMPS}, rerank top-{TASK3_NUCLEAR_RERANK_TOPK} dev gold)")
    if TASK3_SCORE_MAX:
        if TASK3_SCORE_MAX_MODEL.lower() == "gpt-5.2-chat" and not azure_client_gpt52_chat:
            raise RuntimeError(
                "TASK3_SCORE_MAX with gpt-5.2-chat requires Azure client. "
                "Set AZURE_GPT52_CHAT_ENDPOINT, AZURE_GPT52_CHAT_API_VERSION, AZURE_GPT52_CHAT_API_KEY in .env"
            )
        print(f"Score-max mode: on (model={TASK3_SCORE_MAX_MODEL}, candidates={TASK3_SCORE_MAX_CANDIDATES}, temp={TASK3_SCORE_MAX_TEMP})")
    
    # Resume support: load existing predictions so we don't overwrite good results
    existing_preds: Dict[str, str] = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            for item in prev:
                pred = item.get("prediction", "").strip()
                if pred:  # only keep non-empty predictions
                    existing_preds[item["case_id"]] = pred
            if existing_preds:
                print(f"Resume: loaded {len(existing_preds)} existing non-empty predictions from {out_path.name}")
        except Exception:
            pass
    
    submissions = []
    for i, case in enumerate(cases):
        cid = case["case_id"]
        t0 = time.time()
        
        # Skip cases that already have a good prediction (resume)
        if cid in existing_preds:
            answer_text = existing_preds[cid]
            wc = len(answer_text.split())
            print(f"[{i+1}/{len(cases)}] Case {cid} -> CACHED {wc}w")
            submissions.append({"case_id": cid, "prediction": answer_text})
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(submissions, f, indent=2, ensure_ascii=False)
            continue
        
        print(f"[{i+1}/{len(cases)}] Case {cid}")
        evidence_ids = evidence_map.get(cid) if evidence_map else None
        
        # Exclude current case from few-shot; optionally rank by similarity and take top N
        few_shot_for_case = [ex for ex in few_shot_examples_all if ex["case"]["case_id"] != cid]
        if TASK3_SIMILAR_FEW_SHOT and few_shot_for_case:
            few_shot_for_case = _rank_few_shot_by_similarity(few_shot_for_case, case, TASK3_FEW_SHOT_N)
        
        # ========== NUCLEAR PIPELINE (targets all 6 leaderboard metrics) ==========
        if TASK3_NUCLEAR:
            # Build nuclear prompt (predict-clinician-answer framing, all dev gold examples)
            few_shot_nuclear = few_shot_for_case
            if TASK3_SIMILAR_FEW_SHOT and few_shot_nuclear:
                few_shot_nuclear = _rank_few_shot_by_similarity(few_shot_nuclear, case, TASK3_FEW_SHOT_N)
            few_shot_o3_nuclear = few_shot_nuclear[:O3_FEW_SHOT_N] if few_shot_nuclear and len(few_shot_nuclear) > O3_FEW_SHOT_N else few_shot_nuclear

            prompt_full = build_nuclear_prompt(case, evidence_ids, few_shot_nuclear, gold_evidence_map)
            prompt_o3 = build_nuclear_prompt(case, evidence_ids, few_shot_o3_nuclear, gold_evidence_map)

            # Generate multiple candidates across all ensemble models
            raw_answers: List[tuple] = []
            if use_ensemble:
                for dep in ENSEMBLE_DEPLOYMENTS:
                    is_o3 = dep.lower() in ("o3", "o3-pro")
                    prompt = prompt_o3 if is_o3 else prompt_full
                    temps = [0.0] if is_o3 else (TASK3_NUCLEAR_TEMPS[:max(1, TASK3_NUCLEAR_CANDIDATES)] or [0.0])
                    for temp in temps:
                        response = call_azure_chat(prompt, max_tokens=400, temperature=temp, deployment=dep)
                        if response and response.strip():
                            raw_answers.append((dep, temp, response.strip()))
            else:
                for temp in TASK3_NUCLEAR_TEMPS[:max(1, TASK3_NUCLEAR_CANDIDATES)]:
                    response = call_azure_chat(prompt_full, max_tokens=400, temperature=temp)
                    if response and response.strip():
                        raw_answers.append((AZURE_DEPLOYMENT, temp, response.strip()))

            # Process candidates: strip citations, smart truncation, deduplicate
            candidates: List[str] = []
            seen = set()
            for (dep, temp, raw) in raw_answers:
                clean = strip_citations(raw)
                clean = truncate_at_sentence_boundary(clean, MAX_ANSWER_WORDS)
                key = clean.lower().strip()
                if clean and key and key not in seen:
                    seen.add(key)
                    candidates.append(clean)

            print(f"  Nuclear: {len(raw_answers)} raw -> {len(candidates)} unique candidates")

            # Rerank by composite BERTScore + ROUGE vs similar dev gold answers
            if len(candidates) > 1:
                answer_text = _pick_best_nuclear(
                    candidates, case, few_shot_for_case,
                    top_k=TASK3_NUCLEAR_RERANK_TOPK,
                )
            elif candidates:
                answer_text = candidates[0]
            else:
                answer_text = ""

        # ========== SCORE-MAXIMIZER PIPELINE ==========
        elif TASK3_SCORE_MAX:
            # Build prompt with gold few-shot examples
            prompt = build_score_max_prompt(
                case, evidence_ids, few_shot_for_case, gold_evidence_map
            )
            # Generate multiple candidates, pick the best
            candidates = []
            n_cands = TASK3_SCORE_MAX_CANDIDATES
            for ci in range(n_cands):
                temp = TASK3_SCORE_MAX_TEMP if n_cands > 1 else 0.0
                resp = call_azure_chat(
                    prompt, max_tokens=400,
                    temperature=temp,
                    deployment=TASK3_SCORE_MAX_MODEL,
                )
                if resp and resp.strip():
                    text = resp.strip()
                    # Remove any citation markers that slipped in
                    text = strip_citations(text)
                    # Remove date patterns (YYYY-MM-DD, MM/DD/YYYY, etc.)
                    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
                    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    candidates.append(text)
                    print(f"    candidate {ci+1}: {len(text.split())}w")
            
            # Pick best candidate: prefer longest that's ≤75 words and ends properly
            if candidates:
                valid = []
                for c_text in candidates:
                    words = c_text.split()
                    wc = len(words)
                    # Prefer answers in the sweet spot (65-75 words)
                    ends_well = c_text.rstrip()[-1] in '.!?' if c_text.rstrip() else False
                    valid.append((c_text, wc, ends_well))
                
                # Score: prioritize word count (more = better overlap), then sentence completion
                valid.sort(key=lambda x: (
                    x[1] <= MAX_ANSWER_WORDS,  # within limit first
                    x[2],  # ends with punctuation
                    x[1] if x[1] <= MAX_ANSWER_WORDS else -x[1],  # longest within limit
                ), reverse=True)
                answer_text = valid[0][0]
            else:
                answer_text = ""
        
        # ========== FAITHFUL TWO-STAGE PIPELINE ==========
        elif TASK3_FAITHFUL:
            # Stage 1: grounded draft generation (multi-candidate per model if enabled)
            few_shot_o3 = few_shot_for_case[:O3_FEW_SHOT_N] if few_shot_for_case and len(few_shot_for_case) > O3_FEW_SHOT_N else few_shot_for_case
            prompt_full = build_faithful_stage1_prompt(case, evidence_ids, few_shot_for_case, gold_evidence_map)
            prompt_o3 = build_faithful_stage1_prompt(case, evidence_ids, few_shot_o3, gold_evidence_map)

            answers: List[tuple] = []
            if use_ensemble:
                for dep in ENSEMBLE_DEPLOYMENTS:
                    prompt = prompt_o3 if dep.lower() in ("o3", "o3-pro") else prompt_full
                    is_o3 = dep.lower() in ("o3", "o3-pro")
                    temps = [0.0] if is_o3 else (TASK3_SAMPLE_TEMPS[:max(1, TASK3_SAMPLES_PER_MODEL)] or [0.0])
                    for temp in temps:
                        response = call_azure_chat(prompt, max_tokens=400, temperature=temp, deployment=dep)
                        if response and response.strip():
                            answers.append((dep, temp, response.strip()))
            else:
                response = (call_azure_chat(prompt_full, max_tokens=400) or "").strip()
                if response:
                    answers = [(AZURE_DEPLOYMENT, 0.0, response)]
            draft = answers[0][2] if answers else ""

            # Extract cited sentence IDs from first draft (for fallback stage2)
            cited_ids = set()
            for m in re.findall(r'\[(\d+)\]', draft):
                cited_ids.add(m)
            if evidence_ids:
                for eid in evidence_ids:
                    cited_ids.add(eid)
            cited_sentences = [{"id": s["id"], "text": s["text"]} for s in case["sentences"] if s["id"] in cited_ids]

            # New final strategy: rewrite to dev reference style, then rerank by dev gold similarity
            if TASK3_REFERENCE_STYLE_MODE and answers:
                stage1_candidates: List[str] = []
                seen = set()
                for (_, _, d) in answers:
                    cand = truncate_to_max_words(strip_citations(d), MAX_ANSWER_WORDS)
                    key = cand.lower().strip()
                    if cand and key and key not in seen:
                        seen.add(key)
                        stage1_candidates.append(cand)
                note_context = _expanded_context_text_for_case(case, evidence_ids)
                style_examples = few_shot_for_case[:max(1, TASK3_REWRITE_FEW_SHOT_N)]
                if TASK3_SIMILAR_FEW_SHOT and style_examples:
                    style_examples = _rank_few_shot_by_similarity(style_examples, case, max(1, TASK3_REWRITE_FEW_SHOT_N))
                rewritten_candidates: List[str] = []
                for cand in stage1_candidates:
                    rp = build_reference_style_rewrite_prompt(case, cand, note_context, style_examples)
                    rw = (call_azure_chat(rp, max_tokens=300, deployment=TASK3_REWRITE_DEPLOYMENT) or "").strip()
                    final = truncate_to_max_words(strip_citations(rw or cand), MAX_ANSWER_WORDS)
                    if final:
                        rewritten_candidates.append(final)
                if rewritten_candidates:
                    answer_text = _pick_best_candidate_by_dev_refs(rewritten_candidates, case, few_shot_for_case)
                elif stage1_candidates:
                    answer_text = stage1_candidates[0]
                else:
                    answer_text = ""
                print(f"  Ref-style rewrite+rerank: {len(answer_text.split())}w from {len(stage1_candidates)} drafts")
            elif use_ensemble and TASK3_ENSEMBLE_PICK_BEST and len(answers) > 1:
                candidates = [
                    truncate_to_max_words(strip_citations(d), MAX_ANSWER_WORDS)
                    for (_, _, d) in answers
                    if d and strip_citations(d).strip()
                ]
                if len(candidates) > 1:
                    answer_text = _pick_best_candidate_by_bertscore(candidates, case, evidence_ids)
                elif len(candidates) == 1:
                    answer_text = candidates[0]
                else:
                    answer_text = ""
                print(f"  Ensemble pick-best: {len(answer_text.split())}w")
            elif TASK3_SKIP_STAGE2:
                answer_text = strip_citations(draft)
            elif TASK3_STAGE2_MINIMAL_EDIT:
                answer_text = truncate_to_max_words(strip_citations(draft), MAX_ANSWER_WORDS)
                print(f"  Stage 2 minimal-edit: {len(answer_text.split())}w")
            elif cited_sentences:
                stage2_prompt = build_faithful_stage2_prompt(
                    case["patient_question"], case["clinician_question"],
                    cited_sentences, draft
                )
                answer_text = (call_azure_chat(stage2_prompt, max_tokens=250, deployment=TASK3_REFORMULATION_DEPLOYMENT) or "").strip()
                if not answer_text:
                    answer_text = strip_citations(draft)
            else:
                answer_text = strip_citations(draft)
        
        # ========== ORIGINAL PIPELINE ==========
        else:
            prompt_full = build_answer_prompt(case, few_shot_for_case if few_shot_for_case else None, evidence_ids=evidence_ids)
            few_shot_o3 = few_shot_for_case[:O3_FEW_SHOT_N] if few_shot_for_case and len(few_shot_for_case) > O3_FEW_SHOT_N else few_shot_for_case
            prompt_o3 = build_answer_prompt(case, few_shot_o3 if few_shot_o3 else None, evidence_ids=evidence_ids)
            
            if use_ensemble:
                answers = []
                for dep in ENSEMBLE_DEPLOYMENTS:
                    prompt = prompt_o3 if dep.lower() in ("o3", "o3-pro") else prompt_full
                    response = call_azure_chat(prompt, max_tokens=300, deployment=dep)
                    if response and response.strip():
                        answers.append((dep, response.strip()))
                if answers:
                    answer_text = answers[0][1]
                else:
                    answer_text = ""
            else:
                response = call_azure_chat(prompt_full, max_tokens=300)
                answer_text = response.strip() if response else ""
            
            # Citation-then-strip: remove [1], [2], etc. for final output
            if TASK3_CITATION_THEN_STRIP and answer_text:
                answer_text = strip_citations(answer_text)
            
            # Reformulation pass (LAMAR-style)
            if TASK3_REFORMULATION and answer_text and TASK3_REFORMULATION_DEPLOYMENT:
                answer_text = _reformulate_answer(answer_text, case["patient_question"], TASK3_REFORMULATION_DEPLOYMENT)
        
        answer_text = truncate_to_max_words(answer_text, MAX_ANSWER_WORDS)
        
        elapsed = time.time() - t0
        wc = len(answer_text.split())
        print(f"  -> {wc}w ({elapsed:.1f}s)")
        submissions.append({"case_id": cid, "prediction": answer_text})
        # Incremental save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submissions, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Wrote {out_path} ({len(submissions)} cases)")
    return submissions


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent
    data_dir = base.parent / "task2" / "v1.4_Subtask2" / "v1.4"
    argv = (sys.argv[1:] or ["dev"])
    split = argv[0].lower() if argv else "dev"
    limit = None
    if len(argv) >= 2 and argv[1].isdigit():
        limit = int(argv[1])
        print(f"Limit: {limit} cases")
    
    if TASK3_EXP_TAG:
        out_dir = base / f"submission_{TASK3_EXP_TAG}"
    else:
        out_dir = base / "submission"
    key_path = data_dir / "dev" / "archehr-qa_key.json"
    
    # Full 120-case run for Codabench (dev 1-20 + test 21-120)
    if split in ("full", "120", "codabench"):
        dev_xml = data_dir / "dev" / "archehr-qa.xml"
        test_xml = data_dir / "test" / "archehr-qa.xml"
        if not dev_xml.exists() or not test_xml.exists():
            print("Error: dev and test XML required for full run. Check v1.4/dev and v1.4/test.")
            sys.exit(1)
        out_path_120 = out_dir / "submission_120.json"
        print("=== Full 120-case run (dev 1-20 + test 21-120) for Codabench ===")
        print("Stage 1: Dev (1-20)...")
        sub_dev = run_answer_pipeline(dev_xml, out_dir / "_temp_dev.json", key_path=key_path, data_dir=data_dir)
        print(f"  Done: {len(sub_dev)} cases")
        print("Stage 2: Test (21-120)...")
        sub_test = run_answer_pipeline(test_xml, out_dir / "_temp_test.json", key_path=key_path, data_dir=data_dir)
        print(f"  Done: {len(sub_test)} cases")
        merged = sub_dev + sub_test
        merged.sort(key=lambda x: int(x["case_id"]) if x["case_id"].isdigit() else 0)
        out_path_120.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_120, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        # Remove temp files
        (out_dir / "_temp_dev.json").unlink(missing_ok=True)
        (out_dir / "_temp_test.json").unlink(missing_ok=True)
        print(f"=== Merged {len(merged)} cases -> {out_path_120} ===")
        print("Zip submission_120.json as submission.zip (with file named submission.json inside) to submit to Codabench.")
        sys.exit(0)
    
    if split == "test-2026":
        xml_path = data_dir / "test-2026" / "archehr-qa.xml"
    elif split == "test":
        xml_path = data_dir / "test" / "archehr-qa.xml"
    else:
        xml_path = data_dir / "dev" / "archehr-qa.xml"
    if not xml_path.exists():
        xml_path = data_dir / "dev" / "archehr-qa.xml"
    
    custom_dev_out = os.getenv("TASK3_DEV_OUTPUT")
    if split == "test-2026":
        out_path = out_dir / "submission_test2026.json"
    elif split == "test":
        out_path = out_dir / "submission_test.json"
    else:
        if custom_dev_out:
            out_path = Path(custom_dev_out)
            if not out_path.is_absolute():
                out_path = out_dir / out_path.name
        else:
            out_path = out_dir / "submission_dev.json"
    
    run_answer_pipeline(xml_path, out_path, key_path=key_path, data_dir=data_dir, limit=limit)
