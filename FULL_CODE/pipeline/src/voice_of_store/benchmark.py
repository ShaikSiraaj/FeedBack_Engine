"""Compare the official rule baseline and the TF-IDF classifier on one holdout split."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from voice_of_store.features import load_training_data
from voice_of_store.predict import predict_records
from voice_of_store.train import TARGETS, _load_config, _new_classifier, stratified_indices

PREDICTION_COLUMNS = ["feedback_id", "issue_type_id", "severity", "actionable"]
FLAG_COLUMNS = [
    "as_of_week_start",
    "store_id",
    "issue_type_id",
    "confidence",
    "evidence_feedback_ids",
]


def _load_baseline(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("official_r5_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load baseline module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_official_evaluator(
    evaluator: Path,
    predictions: Path,
    truth: Path,
    flags: Path,
    incidents: Path,
    data_dir: Path,
    score_path: Path,
) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--submission",
            str(predictions),
            "--truth",
            str(truth),
            "--flags",
            str(flags),
            "--incidents",
            str(incidents),
            "--feedback",
            str(data_dir / "feedback.csv"),
            "--stores",
            str(data_dir / "store_master.csv"),
            "--out",
            str(score_path),
        ],
        check=True,
    )
    with score_path.open(encoding="utf-8") as file:
        return json.load(file)


def run_benchmark(
    data_dir: Path,
    baseline_path: Path,
    evaluator_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    output_dir = output_dir.resolve()
    protected_dirs = (project_root / "submission", project_root / "FULL_CODE" / "submission")
    inside_submission = any(
        output_dir == protected or protected in output_dir.parents
        for protected in protected_dirs
    )
    if inside_submission:
        raise ValueError(
            f"Benchmark outputs must not be written under a submission directory: {output_dir}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new path: {output_dir}")
    for path in (data_dir, baseline_path, evaluator_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"Required benchmark input does not exist: {path}")
    required_files = (
        "feedback.csv",
        "feedback_labels_train.csv",
        "issue_taxonomy.csv",
        "store_master.csv",
    )
    for name in required_files:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Required dataset is missing: {data_dir / name}")

    _, _, seed = _load_config(config_path)
    labelled, taxonomy = load_training_data(data_dir)
    train_indices, validation_indices = stratified_indices(labelled["issue_type_id"], seed)
    train_rows = labelled.iloc[train_indices]
    validation_rows = labelled.iloc[validation_indices]

    models = {}
    for target in TARGETS:
        model = _new_classifier(seed)
        model.fit(train_rows["text"], train_rows[target])
        models[target] = model
    taxonomy_rules = taxonomy.set_index("issue_type_id")[
        ["owner_function", "is_actionable_category", "policy_severity_floor"]
    ].to_dict(orient="index")
    model_bundle = {"models": models, "taxonomy": taxonomy_rules, "seed": seed}

    feedback = pd.read_csv(data_dir / "feedback.csv", low_memory=False)
    validation_ids = validation_rows["feedback_id"].tolist()
    validation_feedback = (
        feedback.set_index("feedback_id").loc[validation_ids].reset_index()
    )
    baseline = _load_baseline(baseline_path)
    baseline_predictions = baseline.label(
        validation_feedback, taxonomy.set_index("issue_type_id")
    )[PREDICTION_COLUMNS]
    classifier_predictions = predict_records(validation_feedback, model_bundle)[
        PREDICTION_COLUMNS
    ]
    truth = validation_rows[PREDICTION_COLUMNS].copy()

    output_dir.mkdir(parents=True)
    truth_path = output_dir / "validation_truth.csv"
    flags_path = output_dir / "empty_validation_flags.csv"
    incidents_path = output_dir / "empty_validation_incidents.csv"
    truth.to_csv(truth_path, index=False)
    pd.DataFrame(columns=FLAG_COLUMNS).to_csv(flags_path, index=False)
    pd.DataFrame(columns=["incident_id"]).to_csv(incidents_path, index=False)

    metric_names = (
        "n_scored",
        "n_missing_predictions",
        "issue_type_macro_f1",
        "severity_weighted_f1",
        "actionable_f1",
        "classification_score",
    )
    results = {
        "split": {
            "strategy": "80/20 holdout stratified by issue_type_id",
            "seed": seed,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
        },
        "scoring_note": (
            "Official evaluator classification metrics only. No flags/incidents are "
            "scored; do not compare primary_score or lead_score."
        ),
        "models": {},
    }

    for name, predictions in (
        ("official_rule_baseline", baseline_predictions),
        ("tfidf_logistic_regression", classifier_predictions),
    ):
        prediction_path = output_dir / f"{name}_predictions.csv"
        score_path = output_dir / f"{name}_score.json"
        predictions.to_csv(prediction_path, index=False)
        score = _run_official_evaluator(
            evaluator_path,
            prediction_path,
            truth_path,
            flags_path,
            incidents_path,
            data_dir,
            score_path,
        )
        results["models"][name] = {key: score[key] for key in metric_names}

    results_path = output_dir / "benchmark_results.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]
    full_code = project_root / "FULL_CODE"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=full_code / "baseline" / "baseline.py")
    parser.add_argument("--evaluator", type=Path, default=full_code / "evalutor" / "evalute.py")
    parser.add_argument("--config", type=Path, default=full_code / "pipeline" / "config.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = run_benchmark(
        args.data_dir.resolve(),
        args.baseline.resolve(),
        args.evaluator.resolve(),
        args.config.resolve(),
        args.out_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
