# [Data Mining Lab @ Yale] at ArchEHR-QA 2026 — Subtask 3: Answer Generation

This repository contains the **Yale-DM-Lab(https://github.com/Data-Mining-Lab-Yale)** system for **Subtask 3 (Answer Generation)** of the [ArchEHR-QA 2026] shared task (CL4Health @ LREC-COLING 2026). The task studies patient-authored questions about hospitalization records. Subtask 3 generates an answer of at most 75 words grounded in the provided clinical note, in professional register and without citations in the output. Evaluation uses BLEU, ROUGE, SARI, BERTScore (and optionally AlignScore and MEDCON in the official environment; development experiments in this repo use locally reproducible metrics only).

**Note:** **Subtask 2 (Evidence Identification)** is released separately: [ArchEHR-QA-2026-Subtask2](https://github.com/Elyasirankhah/ArchEHR-QA-2026-Subtask2). This repository is for Subtask 3 only.

## Overview

- **Input:** Patient question, clinician-interpreted question, identified evidence sentences (from ST2), and the full clinical note.
- **Output:** A single answer string per case (≤75 words), hard-truncated from the model output. Evaluated with BLEU, ROUGE-Lsum, SARI, and BERTScore (4-metric mean used locally).
- **Approach:** Few-shot prompting with dev examples (5, 10, or 15), optional two-stage *faithful* pipeline (cited draft → rewrite using only the cited sentences), and ensemble of Azure-hosted models (o3, GPT-5.2, GPT-5.1) with cascade fallback.

**Results (aligned with the paper):** The best development score is **34.01** using the o3 + GPT-5.2 + GPT-5.1 ensemble. The faithfulness pipeline (two-stage: cited draft then rewrite) achieves **33.53** on dev with 10 few-shot examples; a 120-case run reaches **33.85**. On the test set, the ensemble reaches **30.95**.

---

## Repository Layout

Use this code as the **`task3`** directory of the full ArchEHR-QA repository so that paths to data and evaluation resolve correctly.

```
your-repo/
├── task2/
│   ├── v1.4_Subtask2/v1.4/   # or v1.5: dev/ and test/ archehr-qa.xml + archehr-qa_key.json
│   └── Evaluation_Task2/evaluation/   # scorers (bleu, rouge, sari, bert)
└── task3/   # <-- contents of THIS release
    ├── pipeline_subtask3_answer.py
    ├── score_minimal.py
    ├── run_approaches.py
    ├── submission/
    └── requirements.txt
```

Run all commands from the **repository root** (parent of `task3`), e.g.:

```bash
python task3/pipeline_subtask3_answer.py dev
```

---

## Prerequisites

- **Python 3.9+**
- **Azure OpenAI** access (o3, GPT-5.2, GPT-5.1) — set API key and endpoint in a `.env` file at the repository root (see Environment variables below).
- **ArchEHR-QA data**: 20 development and 147 test cases; place dev/test `archehr-qa.xml` and `archehr-qa_key.json` under `task2/v1.4_Subtask2/v1.4/dev/` and `.../test/` (or v1.5 equivalent).
- **Task 2 evaluation package**: `task2/Evaluation_Task2/evaluation/` with `scorers/` (BLEU, ROUGE, SARI, BERT) so that `score_minimal.py` can import them.

---

## Installation

From the repository root:

```bash
pip install -r task3/requirements.txt
```

Create a `.env` file at the repository root with your Azure credentials and any optional variables (see below). Do not commit `.env` to version control.

---

## Environment Variables

Set these in a `.env` file at the repository root. Required for the pipeline:

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | API key |
| `AZURE_ENSEMBLE_DEPLOYMENTS` | Comma-separated deployment names (default: `o3,gpt-5.2,gpt-5.1`) |

Optional:

| Variable | Description | Default |
|----------|-------------|---------|
| `TASK3_FEW_SHOT_N` | Number of dev few-shot examples (5, 10, or 15) | `10` |
| `TASK3_FAITHFUL` | Use two-stage faithful pipeline (cited draft → rewrite) | `0` (set `1` for faithfulness variant) |
| `TASK3_USE_TASK2_EVIDENCE` | Restrict note to ST2 evidence sentences | `1` |
| `TASK2_SUBMISSION` | Path to ST2 submission JSON (evidence IDs) | `task2/submission_Subtask2/submission.json` |

---

## How to Run

**Development (20 cases):**

```bash
python task3/pipeline_subtask3_answer.py dev
```

Output: `task3/submission/submission_dev.json`

**Full 120-case run (dev 1–20 + test 21–120) for CodaBench:**

```bash
python task3/pipeline_subtask3_answer.py full
# or: python task3/pipeline_subtask3_answer.py 120
```

Output: `task3/submission/submission_120.json`. Zip it as `submission.zip` with the file named `submission.json` inside for submission.

**Test 2026 (47 cases):**

```bash
python task3/pipeline_subtask3_answer.py test-2026
```

Output: `task3/submission/submission_test2026.json`

**Limit number of cases (e.g. 5):**

```bash
python task3/pipeline_subtask3_answer.py dev 5
```

---

## Scoring

The minimal scorer computes BLEU, ROUGE-Lsum, SARI, and BERTScore. MEDCON and AlignScore are part of the official evaluation but are not reproduced in this minimal scorer:

```bash
python task3/score_minimal.py
```

Uses `task3/submission/submission_dev.json` by default. Options:

- `--submission path/to/submission.json` — score that file  
- `--out path/to/scores.json` — write scores there  
- `--full` — score 120-case submission (dev+test key and sources)  
- `--subset 1,2,3,5` — score only listed case IDs (e.g. tune/hold-out split)

---

## Submission Format

JSON array of objects, one per case: `{"case_id": "1", "prediction": "..."}`. `prediction` must be a single string (≤75 words recommended); no citations in the text.

---

