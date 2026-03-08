"""
Minimal Task 3 scorer: BLEU, ROUGE, SARI, BERTScore only (no QuickUMLS/MEDCON, no AlignScore).
Use this to get comparable scores vs 2025 when full scorer is not available.
Overall (approx) = average of these 4 metrics (same scale as official: 0-100).
2025 best: dev overall 51.5, test overall 49.1 (official = 6 metrics including MEDCON, AlignScore).
"""

import json
import sys
from pathlib import Path
import numpy as np

TASK3_DIR = Path(__file__).resolve().parent
TASK2 = TASK3_DIR.parent / "task2"
EVAL_DIR = TASK2 / "Evaluation_Task2" / "evaluation"

# Add evaluation dir so we can import scorers (they don't import medcon)
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Import only scorers that don't need QuickUMLS/AlignScore
from scorers.bleu_scorer import BleuScorer
from scorers.rouge_scorer import RougeScorer
from scorers.sari_scorer import SariScorer
from scorers.bert_scorer import BertScorer

V14 = TASK2 / "v1.4_Subtask2" / "v1.4"
SUBMISSION = TASK3_DIR / "submission" / "submission_dev.json"
KEY = V14 / "dev" / "archehr-qa_key.json"
KEY_TEST = V14 / "test" / "archehr-qa_key.json"
DATA = V14 / "dev" / "archehr-qa.xml"
DATA_TEST = V14 / "test" / "archehr-qa.xml"
OUT = TASK3_DIR / "scores_dev.json"
OUT_120 = TASK3_DIR / "scores_120.json"


def get_args():
    import argparse
    p = argparse.ArgumentParser(description="Score Task 3 submission (4 metrics).")
    p.add_argument("--submission", type=str, default=None, help="Path to submission JSON")
    p.add_argument("--out", type=str, default=None, help="Path to write scores JSON")
    p.add_argument("--full", "--120", dest="full", action="store_true", help="Score 120 cases: use dev+test key and sources")
    p.add_argument("--subset", type=str, default=None, help="Score only these case IDs (comma-separated, e.g. 1,2,3,5 for tune or 4,9,14,19 for hold-out)")
    return p.parse_args()


def load_submission(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"case_id": c["case_id"], "prediction": c["prediction"].strip()} for c in data]


def load_key(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["case_id"]: c["clinician_answer_without_citations"].strip() for c in data}


def load_sources(data_path):
    from lxml import etree
    tree = etree.parse(data_path)
    root = tree.getroot()
    out = {}
    for case_el in root.findall(".//case"):
        case_id = case_el.get("id", "")
        patient_el = case_el.find("patient_narrative")
        if patient_el is not None and patient_el.text is not None:
            out[case_id] = patient_el.text.strip()
        else:
            out[case_id] = ""
    return out


def load_key_120():
    """Merge dev + test key for 120 cases (1-20 + 21-120)."""
    key_map = load_key(KEY)
    if KEY_TEST.exists():
        key_map = {**key_map, **load_key(KEY_TEST)}
    return key_map


def load_sources_120():
    """Merge dev + test XML sources for SARI (patient_narrative)."""
    source_map = load_sources(DATA)
    if DATA_TEST.exists():
        source_map = {**source_map, **load_sources(DATA_TEST)}
    return source_map


def compute_scores(submission_path=None, out_path=None, quiet=False, full_120=False, subset_case_ids=None):
    """Compute and return scores dict. If out_path set, also write there.
    full_120: use dev+test key and sources for 120-case submission.
    subset_case_ids: optional set of case_id strings; score only those cases (e.g. tune or hold-out)."""
    submission_path = Path(submission_path) if submission_path else SUBMISSION
    out_path = Path(out_path) if out_path else (OUT_120 if full_120 else OUT)
    if not quiet:
        print("Loading submission...")
    submission = load_submission(submission_path)
    if subset_case_ids is not None:
        submission = [c for c in submission if c["case_id"] in subset_case_ids]
        if not quiet:
            ids = sorted([c["case_id"] for c in submission], key=lambda x: (int(x) if str(x).isdigit() else 0))
            print(f"  Subset: {len(submission)} cases {ids}")
    if not quiet:
        print(f"  Cases: {len(submission)}")
    if full_120:
        if not quiet:
            print("Loading key (dev + test)...")
        key_map = load_key_120()
        if not quiet:
            print("Loading sources (dev + test, for SARI)...")
        source_map = load_sources_120()
    else:
        if not quiet:
            print("Loading key...")
        key_map = load_key(KEY)
        if not quiet:
            print("Loading sources (for SARI)...")
        source_map = load_sources(DATA)

    refs = []
    preds = []
    srcs = []
    missing = []
    empty_preds = []
    for c in submission:
        cid = c["case_id"]
        if cid not in key_map:
            missing.append(cid)
            continue
        pred = (c.get("prediction") or "").strip()
        if not pred:
            empty_preds.append(cid)
            continue
        refs.append(key_map[cid])
        preds.append(pred)
        srcs.append(source_map.get(cid, ""))
    if missing and not quiet:
        print(f"  Warning: {len(missing)} cases not in key (skipped): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if empty_preds and not quiet:
        print(f"  Warning: {len(empty_preds)} case(s) with empty prediction (skipped for scoring): {empty_preds}")
    if not refs or not preds:
        if not quiet:
            print("No valid (ref, pred) pairs; cannot compute scores.")
        return {}

    if not quiet:
        print("\nComputing metrics (BLEU, ROUGE, SARI, BERTScore)...")
    scores = {}

    if not quiet:
        print("  BLEU...")
    s = BleuScorer().compute_overall_score(refs, preds)
    scores["bleu"] = s * 100
    if not quiet:
        print(f"    {scores['bleu']:.2f}")

    if not quiet:
        print("  ROUGE...")
    r = RougeScorer().compute_overall_score(refs, preds)
    scores["rouge1"] = r["rouge1"] * 100
    scores["rouge2"] = r["rouge2"] * 100
    scores["rougeL"] = r["rougeL"] * 100
    scores["rougeLsum"] = r["rougeLsum"] * 100
    if not quiet:
        print(f"    rougeLsum: {scores['rougeLsum']:.2f}")

    if not quiet:
        print("  SARI...")
    try:
        sari = SariScorer().compute_overall_score(refs, preds, srcs)
        scores["sari"] = sari
        if not quiet:
            print(f"    {scores['sari']:.2f}")
        has_sari = True
    except Exception as e:
        if not quiet:
            print(f"    (skip: {e})")
        scores["sari"] = None
        has_sari = False

    if not quiet:
        print("  BERTScore...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = BertScorer(device=device).compute_overall_score(refs, preds)
    scores["bertscore"] = bs * 100
    if not quiet:
        print(f"    {scores['bertscore']:.2f}")

    # Approximate overall = average of available metrics (same 0–100 scale as official)
    # Official overall = mean(bleu, rougeLsum, sari, bertscore, alignscore, medcon)
    vals = [scores["bleu"], scores["rougeLsum"], scores["bertscore"]]
    if scores.get("sari") is not None:
        vals.append(scores["sari"])
    overall_approx = np.mean(vals)
    scores["overall_score_approx"] = overall_approx
    scores["metrics_used"] = "bleu, rougeLsum, bertscore" + (", sari" if scores.get("sari") is not None else "")

    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    if not quiet:
        print("\n" + "=" * 60)
        print("Task 3 " + ("120-case " if full_120 else "dev ") + "scores (4 metrics, no MEDCON/AlignScore)")
        print("=" * 60)
        for k, v in sorted(scores.items()):
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")
        print("\nComparison to 2025 (LAMAR):")
        print("  2025 dev overall (6 metrics): 51.5")
        print("  2025 test overall (6 metrics): 49.1")
        print(f"  Ours overall_approx (4 metrics): {overall_approx:.2f}")
        print(f"Scores saved to: {out_path}")
    return scores


def main():
    args = get_args()
    sub_path = Path(args.submission) if args.submission else SUBMISSION
    out_path = Path(args.out) if args.out else None
    subset = None
    if getattr(args, "subset", None):
        subset = set(x.strip() for x in args.subset.split(",") if x.strip())
    compute_scores(submission_path=sub_path, out_path=out_path, quiet=False, full_120=args.full, subset_case_ids=subset)


if __name__ == "__main__":
    main()
