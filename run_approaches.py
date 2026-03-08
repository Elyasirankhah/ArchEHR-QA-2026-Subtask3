"""
Run Task 3 approaches: pipeline -> score -> print to terminal, save ONLY scores to approach_scores.json.
Does NOT save dev submission output (uses temp file then deletes).
"""
import os
import sys
import json
import subprocess
from pathlib import Path

try:
    from tqdm import tqdm
    _write = tqdm.write
except ImportError:
    def tqdm(iterable, desc="", total=None, **kwargs):
        return iterable
    _write = print

TASK3_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK3_DIR.parent
TEMP_SUBMISSION = TASK3_DIR / "submission" / "submission_dev_temp.json"
TEMP_SCORES = TASK3_DIR / "scores_temp.json"
SCORES_ONLY_FILE = TASK3_DIR / "approach_scores.json"

# Single approach: improve the ~28 (fewshot_10) pipeline to 34+.
# Pipeline defaults: TASK3_FEW_SHOT_N=10, evidence-first, reformulation, ICML-style.
APPROACHES = [
    ("task3_34plus", {}),  # 10 few-shot + Task2 evidence + reformulation + ICML prompt
]


def run_pipeline(env_extra):
    env = {**os.environ, "TASK3_DEV_OUTPUT": TEMP_SUBMISSION.name, **env_extra}
    out = subprocess.run(
        [sys.executable, str(TASK3_DIR / "pipeline_subtask3_answer.py"), "dev"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600000,
    )
    if out.returncode != 0:
        print(out.stderr or out.stdout)
        raise RuntimeError(f"Pipeline failed: {out.returncode}")
    return True


def run_scorer():
    out = subprocess.run(
        [
            sys.executable,
            str(TASK3_DIR / "score_minimal.py"),
            "--submission", str(TEMP_SUBMISSION),
            "--out", str(TEMP_SCORES),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180000,
    )
    if out.returncode != 0:
        print(out.stderr or out.stdout)
        raise RuntimeError("Scorer failed")
    with open(TEMP_SCORES, "r") as f:
        return json.load(f)


def main():
    all_scores = {}
    if SCORES_ONLY_FILE.exists():
        with open(SCORES_ONLY_FILE, "r") as f:
            all_scores = json.load(f)

    n = len(APPROACHES)
    for idx, (name, env_extra) in enumerate(tqdm(APPROACHES, desc="Approaches", unit="approach", total=n), start=1):
        _write("\n" + "=" * 60)
        _write(f" [{idx}/{n}] Approach: {name}")
        _write("=" * 60)
        try:
            run_pipeline(env_extra)
            if not TEMP_SUBMISSION.exists():
                _write("  [WARN] No temp submission produced")
                continue
            scores = run_scorer()
            # Keep only numeric scores for storage (no metrics_used string in summary)
            summary = {
                "overall_score_approx": round(scores["overall_score_approx"], 2),
                "bleu": round(scores["bleu"], 2),
                "rougeLsum": round(scores["rougeLsum"], 2),
                "sari": round(scores["sari"], 2) if scores.get("sari") is not None else None,
                "bertscore": round(scores["bertscore"], 2),
            }
            all_scores[name] = summary
            _write(f"  overall_score_approx: {summary['overall_score_approx']}")
            _write(f"  bleu: {summary['bleu']}  rougeLsum: {summary['rougeLsum']}  sari: {summary['sari']}  bertscore: {summary['bertscore']}")
        except Exception as e:
            _write(f"  ERROR: {e}")
            all_scores[name] = {"error": str(e)}
        finally:
            if TEMP_SUBMISSION.exists():
                TEMP_SUBMISSION.unlink()
            if TEMP_SCORES.exists():
                TEMP_SCORES.unlink()
        # Save after each approach so partial results persist
        with open(SCORES_ONLY_FILE, "w") as f:
            json.dump(all_scores, f, indent=2)

    with open(SCORES_ONLY_FILE, "w") as f:
        json.dump(all_scores, f, indent=2)
    _write("\n" + "=" * 60)
    _write(f" Scores saved to: {SCORES_ONLY_FILE}")
    _write("=" * 60)


if __name__ == "__main__":
    main()
