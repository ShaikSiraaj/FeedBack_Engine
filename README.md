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

This installs the runtime dependencies (`numpy`, `pandas`, `PyYAML`) and the dev tools (`pytest`, `ruff`) from `pyproject.toml` and the committed `uv.lock`.

### 3. Put the organiser data in place

The R5 pack is downloaded separately. Create `FULL_CODE/data/` and copy **only these six CSV files** into it:

```text
FULL_CODE/data/
├── feedback.csv
├── feedback_labels_train.csv
├── emerging_incidents_train.csv
├── issue_taxonomy.csv
├── store_master.csv
└── week_calendar.csv
```

Do not commit these files. `data/` is listed in `.gitignore`, and the same applies to any nested `data/` folder. Do not copy the pack's `evaluate.py`, `baseline/`, `sample_submission/`, or README into the data folder.

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

### 7. Detector interface (not implemented yet)

The required R5 detector/re-run interface, once implemented, is:

```bash
python FULL_CODE/pipeline/src/voice_of_store/predict.py \
  --feedback <path_to_feedback.csv> \
  --as-of <YYYY-MM-DD> \
  --out <flags.csv>
```

`FULL_CODE/pipeline/src/voice_of_store/predict.py` currently provides reusable
classification inference helpers, not this detector CLI. The interface above
is documentation only and is not executable yet.

The pipeline must write these exact files into `submission/`:

- `submission/R5_predictions.csv`
- `submission/R5_emerging_flags.csv`
- `submission/detector_cmd.txt`

## How to run the tests

```bash
cd FULL_CODE/pipeline
uv run pytest
```

`FULL_CODE/pipeline/tests/test_core.py` contains tests for data validation,
label validation, feedback joins, missing text, and taxonomy policy floors.

## Architecture

```text
FULL_CODE/
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
├── db/                        optional schema/migrations + seed script
├── web/                       optional demo UI (calls the API)
├── data/                      local organiser CSVs (ignored)
└── models/                    local downloaded models (ignored)
```

Dependency direction: `web` → `api` → `pipeline` → libraries. The core pipeline must remain importable without a web framework so it can be tested on its own. `api/`, `db/`, and `web/` are currently empty placeholders; they should be removed if they are not part of the final demo.

## Detector command and causality

The required one-line `submission/detector_cmd.txt` must have exactly this shape:

```text
python FULL_CODE/pipeline/src/voice_of_store/predict.py --feedback {feedback} --as-of {asof} --out {out}
```

Rules the detector must follow:

- it runs from the project root, where `README.md`, `submission/`, and `FULL_CODE/` live;
- it writes the flags for the week containing `--as-of` to `--out`;
- every week's flags use only feedback with `created_at` up to that week's Sunday 23:59:59 — never future data;
- it reads `feedback.csv`, `feedback_labels_train.csv`, `store_master.csv`, `issue_taxonomy.csv`, and `week_calendar.csv` from the folder that contains `--feedback`;
- it never hard-codes dates, week numbers, store ids, or feedback ids;
- each flag cites at least two qualifying evidence ids from the same store, week, and issue type.

The submitted `submission/R5_emerging_flags.csv` must be exactly what this command produces, week by week, on the real data.

## Submission files and schemas

### `submission/R5_predictions.csv`

```text
feedback_id,issue_type_id,severity,actionable
```

- one row for every `feedback_id` in `FULL_CODE/data/feedback.csv`;
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
python <R5_PACK>/evaluate.py --submission submission/R5_predictions.csv --truth FULL_CODE/data/feedback_labels_train.csv --feedback FULL_CODE/data/feedback.csv --stores FULL_CODE/data/store_master.csv
```

The evaluator writes `score.json` in the current directory; `score.json` is ignored by git. `find_incidents()` looks for `emerging_incidents_train.csv` next to the truth file, which is why the six CSVs must be together in `FULL_CODE/data/`.

Do not change `evaluate.py`. It is the organiser's scoring tool.

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
