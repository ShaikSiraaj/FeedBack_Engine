import pandas as pd
import pytest

from voice_of_store.features import (
    DataValidationError,
    load_training_data,
    validate_labels,
    validate_taxonomy,
)
from voice_of_store.predict import apply_policy_severity_floor


@pytest.fixture
def taxonomy():
    return pd.DataFrame(
        [
            {
                "issue_type_id": "IT01",
                "owner_function": "Store Ops",
                "policy_severity_floor": "high",
                "is_actionable_category": 1,
            },
            {
                "issue_type_id": "NI01",
                "owner_function": "None (not actionable)",
                "policy_severity_floor": "low",
                "is_actionable_category": 0,
            },
        ]
    )


@pytest.fixture
def labels():
    return pd.DataFrame(
        [
            {
                "feedback_id": "F1",
                "issue_type_id": "IT01",
                "severity": "high",
                "actionable": 0,
            },
            {
                "feedback_id": "F2",
                "issue_type_id": "NI01",
                "severity": "low",
                "actionable": 1,
            },
        ]
    )


def test_taxonomy_requires_unique_ids_and_valid_floor(taxonomy):
    duplicate = pd.concat([taxonomy, taxonomy.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate identifiers"):
        validate_taxonomy(duplicate)

    invalid = taxonomy.copy()
    invalid.loc[0, "policy_severity_floor"] = "urgent"
    with pytest.raises(DataValidationError, match="Unknown policy severity floors"):
        validate_taxonomy(invalid)


def test_taxonomy_requires_observed_schema():
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_taxonomy(pd.DataFrame({"issue_type_id": ["IT01"]}))


def test_labels_must_use_taxonomy_ids_and_valid_targets(taxonomy, labels):
    invalid_issue = labels.copy()
    invalid_issue.loc[0, "issue_type_id"] = "IT99"
    with pytest.raises(DataValidationError, match="Unknown issue_type_id"):
        validate_labels(invalid_issue, taxonomy)

    invalid_actionable = labels.copy()
    invalid_actionable.loc[0, "actionable"] = 2
    with pytest.raises(DataValidationError, match="Unknown actionable values"):
        validate_labels(invalid_actionable, taxonomy)


def test_label_severity_cannot_be_below_taxonomy_floor(taxonomy, labels):
    below_floor = labels.copy()
    below_floor.loc[0, "severity"] = "medium"
    with pytest.raises(DataValidationError, match="below policy severity floors"):
        validate_labels(below_floor, taxonomy)


def test_actionable_target_is_not_overridden_by_category_metadata(taxonomy, labels):
    validated = validate_labels(labels, taxonomy)
    assert validated["actionable"].tolist() == [0, 1]


def test_missing_feedback_text_is_normalized_and_join_is_validated(tmp_path, taxonomy, labels):
    feedback = pd.DataFrame(
        [
            {
                "feedback_id": "F1",
                "channel": "review",
                "created_at": "2026-03-30",
                "store_id": "S1",
                "rating": None,
                "text": None,
            },
            {
                "feedback_id": "F2",
                "channel": "review",
                "created_at": "2026-03-31",
                "store_id": "S1",
                "rating": 3,
                "text": "Feedback text",
            },
        ]
    )
    feedback.to_csv(tmp_path / "feedback.csv", index=False)
    labels.to_csv(tmp_path / "feedback_labels_train.csv", index=False)
    taxonomy.to_csv(tmp_path / "issue_taxonomy.csv", index=False)

    joined, _ = load_training_data(tmp_path)
    assert len(joined) == 2
    assert joined["text"].tolist() == ["", "Feedback text"]

    labels.loc[1, "feedback_id"] = "MISSING"
    labels.to_csv(tmp_path / "feedback_labels_train.csv", index=False)
    with pytest.raises(DataValidationError, match="do not match feedback records"):
        load_training_data(tmp_path)


def test_policy_floor_raises_severity_without_lowering_predictions(taxonomy):
    rules = taxonomy.set_index("issue_type_id").to_dict(orient="index")
    predictions = apply_policy_severity_floor(
        pd.Series(["IT01", "IT01", "NI01"]),
        pd.Series(["low", "critical", "medium"]),
        rules,
    )
    assert predictions.tolist() == ["high", "critical", "medium"]
