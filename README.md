# TEAM_ID-001392 · R5: Voice-of-Store Engine

Turn messy, multilingual store feedback into a prioritised weekly action list routed to the right owner, and flag emerging store-level issues before they become chain-wide.

> **Repository status:** Member 1's first TF-IDF / Logistic Regression classification baseline is implemented with reusable inference helpers. Routing, the emerging-issue detector, regional briefs, and the three required files in `submission/` are not implemented/generated yet. The interfaces and schemas below are fixed by the official R5 problem pack.

## What it does

Saffron Retail Group receives feedback from online reviews, in-app comments, staff shift notes, and call-centre transcripts. The engine will:

- classify each record into an issue type (`IT01`–`IT30` / `NI01`–`NI03`);
- assign a severity (`low`, `medium`, `high`, `critical`) and an actionable flag (`0`/`1`);
- route each issue to its owning function (Store Ops, Supply Chain, Merchandising, Digital, HR);
- aggregate the results into a weekly one-page brief per region;
- detect emerging issues at a single store before they become chain-wide;
- report flag precision and a precision statement for escalated issues.

Member 1 owns the classification and routing pipeline in `FULL_CODE/pipeline/`.

## How to run

### 1. Prerequisites

- Python >= 3.9
- [`uv`](https://docs.astral.sh/uv/) for the committed lockfile

### 2. Install dependencies

```bash
cd FULL_CODE/pipeline
uv sync --extra dev
```

This installs the runtime dependencies (`numpy`, `joblib`, `pandas`, `PyYAML`,
and `scikit-learn`) and the dev tools (`pytest`, `ruff`) from `pyproject.toml`
and the committed `uv.lock`.

### 3. Put the organiser data in place

The checked-in dataset is kept in `FULL_CODE/db/`. Keep these six CSVs together
in the configured data directory:

```text
FULL_CODE/db/
├── feedback.csv
├── feedback_labels_train.csv
├── emerging_incidents_train.csv
├── issue_taxonomy.csv
├── store_master.csv
└── week_calendar.csv
```

Do not copy the pack's `evaluate.py`, `baseline/`, `sample_submission/`, or README into the data folder.

### 4. Models

Model bundles are written locally under `FULL_CODE/models/` (ignored by git).
The baseline does not download pretrained models. If one is added later:

- put downloaded model files under `FULL_CODE/models/` (ignored by git);
- document the exact download command and source in this README;
- keep large downloaded weights out of git;
- a small model trained by the team may be committed inside `FULL_CODE/`, as R5 allows.

### 5. Environment variables

Copy `FULL_CODE/.env.example` to `FULL_CODE/.env` and fill in real values locally. Never commit `FULL_CODE/.env`. The committed `FULL_CODE/.env.example` contains dummy placeholders only.

### 6. Train the classification baseline

The training command validates the feedback, label, and taxonomy schemas; joins
training labels to feedback by `feedback_id`; and evaluates three TF-IDF /
Logistic Regression classifiers (`issue_type_id`, `severity`, and `actionable`).
It uses a reproducible 80/20 holdout stratified by `issue_type_id`, reports
accuracy and macro-F1, then fits the saved inference bundle on all labelled
training rows. Text vectorization is fitted only on the training partition
during evaluation.

```bash
uv run --project FULL_CODE/pipeline python -m voice_of_store.train \
  --config FULL_CODE/pipeline/config.yaml
```

Taxonomy severity floors are applied to predicted severities at inference.
The taxonomy's `is_actionable_category` is descriptive metadata, not a hard
prediction rule: labelled `actionable` values differ from it in the provided
training data, so actionability is trained as an independent target.

### 7. Detector re-run interface

`FULL_CODE/detect.py` is the canonical detector entrypoint. Run the command
from the project root, using the pipeline's installed environment:

```bash
uv run --project FULL_CODE/pipeline python FULL_CODE/detect.py \
  --feedback FULL_CODE/db/feedback.csv \
  --as-of <YYYY-MM-DD> \
  --out <new-temp-directory>/R5_emerging_flags.csv
```

The detector loads the saved classifier from `FULL_CODE/models/`, reads the
supporting `store_master.csv`, `issue_taxonomy.csv`, and `week_calendar.csv`
from the directory containing `--feedback`, and emits the flags for the
calendar week containing `--as-of`. It uses feedback through that week's
Sunday 23:59:59, applies `predict_records` (including policy issue overrides
and severity floors), requires at least two distinct evidence records, and
limits output to three flags for the week. The output columns are exactly
`as_of_week_start,store_id,issue_type_id,confidence,evidence_feedback_ids`;
evidence IDs are semicolon-separated.

The rerun command recorded in `submission/detector_cmd.txt` is generated
without producing submission outputs:

```bash
python FULL_CODE/write_detector_cmd.py --out submission/detector_cmd.txt
```

Its one-line contents are:

```text
uv run --project FULL_CODE/pipeline python FULL_CODE/detect.py --feedback {feedback} --as-of {asof} --out {out}
```

The placeholders are supplied by the rerun caller. The recorded command is
relative to the project root and uses the dependencies installed above.

Final submission generation is not implemented yet. When generated, the
required files in the root-level `submission/` are:

- `submission/R5_predictions.csv`
- `submission/R5_emerging_flags.csv`
- `submission/detector_cmd.txt`

## How to run the tests

```bash
cd FULL_CODE/pipeline
uv run pytest
```

The tests cover data and label validation, policy floors, detector evidence
validity against the official evaluator, flag limits, and the detector
re-run command contract.

## Architecture

```text
FULL_CODE/
├── detect.py                  canonical emerging-incident detector CLI
├── write_detector_cmd.py      writes the detector re-run command template
├── db/                         organiser CSVs used by this checkout
├── models/                     saved classifier bundle (ignored)
├── pipeline/                  core logic; no web-framework imports
│   ├── pyproject.toml         dependencies + tool config
│   ├── uv.lock                committed lockfile
│   ├── config.yaml            relative paths, seed, future thresholds/model names
│   ├── src/voice_of_store/
│   │   ├── features.py        data loading, validation, and text preparation
│   │   ├── train.py           validation and classifier training
│   │   └── predict.py         classification inference helpers
│   └── tests/                 core tests
├── api/                       optional HTTP layer (imports pipeline, never the reverse)
└── web/                        optional demo UI (calls the API)
```

Dependency direction: `web` → `api` → `pipeline` → libraries. The core pipeline must remain importable without a web framework so it can be tested on its own. `api/` and `web/` are optional layers.

## Detector command and causality

The required one-line `submission/detector_cmd.txt`, generated by
`FULL_CODE/write_detector_cmd.py`, has this shape:

```text
uv run --project FULL_CODE/pipeline python FULL_CODE/detect.py --feedback {feedback} --as-of {asof} --out {out}
```

Rules the detector must follow:

- it runs from the project root, where `README.md`, `submission/`, and `FULL_CODE/` live;
- it writes flags for the calendar week containing `--as-of` to `--out`, using feedback through that week's Sunday 23:59:59;
- it reads the supplied `feedback.csv` plus `store_master.csv`, `issue_taxonomy.csv`, and `week_calendar.csv` from the folder that contains `--feedback`;
- it never hard-codes dates, week numbers, store ids, or feedback ids;
- each flag cites at least two distinct evidence IDs from the same store, week, and predicted issue type, separated by semicolons;
- it emits no more than three flags for the week.

The submitted `submission/R5_emerging_flags.csv` must be exactly what this command produces, week by week, on the real data.

## Submission files and schemas

### `submission/R5_predictions.csv`

```text
feedback_id,issue_type_id,severity,actionable
```

- one row for every `feedback_id` in `FULL_CODE/db/feedback.csv`;
- no duplicate ids;
- `issue_type_id` in `IT01`–`IT30`, `NI01`–`NI03`;
- `severity` in `low`, `medium`, `high`, `critical`;
- `actionable` in `0`, `1`.

### `submission/R5_emerging_flags.csv`

```text
as_of_week_start,store_id,issue_type_id,confidence,evidence_feedback_ids
```

- zero or more rows; the header row is always required;
- one row per `(as_of_week_start, store_id, issue_type_id)`;
- `confidence` in `[0, 1]`;
- `evidence_feedback_ids` is a `;`-separated list.

### `submission/detector_cmd.txt`

The exact one-line command shown in "Detector command and causality".

These three files are **not generated yet**. They are produced by the pipeline; do not hand-edit them. `submission/` currently contains only `.gitkeep`.

## Official evaluator

The official `evaluate.py` is part of the downloaded R5 pack and is **not modified or copied into this repository**. Run it from the project root with a Python environment that has `pandas` and `numpy` (the pipeline virtual environment works), against the files produced here:

```bash
python <R5_PACK>/evaluate.py --submission submission/R5_predictions.csv --truth FULL_CODE/db/feedback_labels_train.csv --feedback FULL_CODE/db/feedback.csv --stores FULL_CODE/db/store_master.csv
```

The evaluator writes `score.json` in the current directory; `score.json` is ignored by git. `find_incidents()` looks for `emerging_incidents_train.csv` next to the truth file, which is why the six CSVs must be together in `FULL_CODE/db/`.

Do not change `evaluate.py`. It is the organiser's scoring tool.

### Fair classifier benchmark

The official baseline is at `FULL_CODE/baseline/baseline.py`. Its defaults do not
match this checkout: `ROOT` resolves to `FULL_CODE`, where it expects data under
`FULL_CODE/data/` and writes full-run outputs under `FULL_CODE/submission/`.
The checked-in dataset is in `FULL_CODE/db/`; the final submission directory is
the separate root-level `submission/`. The benchmark below passes the database
directory explicitly and calls the rule classifier in memory, so neither
submission directory is read or written.

The pipeline config's `data_dir: ../db` resolves to `FULL_CODE/db/` in this
checkout. The benchmark uses that directory and the config's seed and existing 80/20 issue-type-stratified
split. For the TF-IDF classifier, it fits temporary models only on that split's
training rows and applies the same inference helper (including severity floors);
it does not read or overwrite `FULL_CODE/models/r5_classifier.joblib`.

The official scorer evaluates the two prediction files against the same
validation-only truth file and reports issue-type macro-F1, severity
weighted-F1, actionable F1, and its classification score. Flags and incidents
are empty for this classification-only comparison, so do not compare
`primary_score` or `lead_score`. Outputs are written to a new benchmark-only
directory; the command refuses to overwrite an existing directory.

From the project root in PowerShell, use a fresh output directory name for each
run:

```powershell
.\FULL_CODE\pipeline\.venv\Scripts\python.exe .\FULL_CODE\pipeline\src\voice_of_store\benchmark.py `
  --data-dir .\FULL_CODE\db `
  --baseline .\FULL_CODE\baseline\baseline.py `
  --evaluator .\FULL_CODE\evalutor\evalute.py `
  --config .\FULL_CODE\pipeline\config.yaml `
  --out-dir .\FULL_CODE\pipeline\benchmark-results\run-2026-10-27
```

The run directory contains the shared validation truth, each model's
validation-only predictions and official evaluator JSON, plus
`benchmark_results.json` with the directly comparable classification metrics.

## Packaging

The submission is one zip named `TEAM_ID-001392_R5.zip`, made from a fresh clone:

```bash
git clone <repo-url> TEAM_ID-001392_R5
cd TEAM_ID-001392_R5
# verify: README.md, .gitignore, LICENSE, submission/, FULL_CODE/ only
# zip the whole clone, including the hidden .git folder
```

Before zipping, confirm that `data/`, `models/`, `.env`, `.venv/`, `node_modules/`, caches, and build output are absent. They are covered by `.gitignore`.

## AI usage

- Runtime AI/LLM usage: none. Classification uses the locally trained TF-IDF and Logistic Regression baseline; no LLM API, agent, or downloaded model is wired in.
- AI coding assistant used for this structure/setup pass: OpenCode.
- Update this section when the team adds an LLM, embedding model, agent, or external AI API.

## Pre-built and third-party code

- Application code from other projects: none.
- Code written by the team before Day 1: only this project skeleton.
- Organiser-provided R5 pack (problem README, `evaluate.py`, `baseline/baseline.py`, sample submission, synthetic data): used locally for development and scoring; not copied into this repository.
- Third-party libraries: `numpy`, `pandas`, `PyYAML`, `scikit-learn` (runtime); `pytest`, `ruff` (development). Exact versions are locked in `uv.lock`.
- Data licence: the R5 synthetic data is CC BY 4.0, as stated in the pack. Project code is MIT; see `LICENSE`.

## Limitations

- The TF-IDF / Logistic Regression classifier baseline and reusable inference helpers are implemented. Routing, the regional brief, and emerging-issue detector are not implemented yet.
- The three required `submission/` files are not generated yet.
- Validation metrics are from a single stratified holdout and are not a substitute for official evaluation.
- `api/`, `db/`, and `web/` are empty placeholders; no API, database schema, or UI exists.
- No de-duplication across channels or precision statement has been produced yet.
- `uv.lock` fixes library versions, but Python 3.9+ compatibility still needs to be tested on the judging machine.
