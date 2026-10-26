"""Train and validate the R5 text-classification baseline."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from voice_of_store.features import load_training_data

TARGETS = ("issue_type_id", "severity", "actionable")


def _new_classifier(seed: int):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, random_state=seed),
    )


def stratified_indices(labels: pd.Series, seed: int) -> tuple[Any, Any]:
    """Create a reproducible 80/20 split stratified by issue type."""
    indices = range(len(labels))
    return train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )


def train_and_evaluate(
    data_dir: Path, seed: int
) -> tuple[dict[str, Any], dict[str, dict[str, float]], int, int]:
    """Evaluate on a held-out split, then fit final models on all labelled rows."""
    labelled, taxonomy = load_training_data(data_dir)
    train_indices, validation_indices = stratified_indices(labelled["issue_type_id"], seed)

    text = labelled["text"]
    validation_metrics: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        validation_model = _new_classifier(seed)
        validation_model.fit(text.iloc[train_indices], labelled[target].iloc[train_indices])
        predicted = validation_model.predict(text.iloc[validation_indices])
        expected = labelled[target].iloc[validation_indices]
        validation_metrics[target] = {
            "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        }

    models = {}
    for target in TARGETS:
        model = _new_classifier(seed)
        model.fit(text, labelled[target])
        models[target] = model

    taxonomy_rules = taxonomy.set_index("issue_type_id")[
        ["owner_function", "is_actionable_category", "policy_severity_floor"]
    ].to_dict(orient="index")
    bundle = {"models": models, "taxonomy": taxonomy_rules, "seed": seed}
    return bundle, validation_metrics, len(train_indices), len(validation_indices)


def _load_config(config_path: Path) -> tuple[Path, Path, int]:
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    base_dir = config_path.parent
    data_dir = (base_dir / config["data_dir"]).resolve()
    models_dir = (base_dir / config["models_dir"]).resolve()
    seed = config["seed"]
    if not isinstance(seed, int):
        raise ValueError("Configuration seed must be an integer")
    return data_dir, models_dir, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[2] / "config.yaml",
        help="Path to pipeline config.yaml",
    )
    args = parser.parse_args()
    data_dir, models_dir, seed = _load_config(args.config)
    bundle, metrics, train_rows, validation_rows = train_and_evaluate(data_dir, seed)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "r5_classifier.joblib"
    joblib.dump(bundle, model_path)
    report = {
        "split": {
            "strategy": "80/20 holdout stratified by issue_type_id",
            "seed": seed,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
        },
        "validation_metrics": metrics,
        "model_bundle": str(model_path),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
