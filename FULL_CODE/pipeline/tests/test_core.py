import pandas as pd
import pytest

from voice_of_store.features import (
    DataValidationError,
    load_training_data,
    validate_labels,
    validate_taxonomy,
)
from voice_of_store.policy_rules import apply_explicit_policy_rules
from voice_of_store.predict import apply_policy_severity_floor, predict_records


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


def test_policy_floor_covers_every_floor_bearing_category():
    floors = {
        "IT04": "high",
        "IT06": "high",
        "IT08": "high",
        "IT10": "high",
        "IT16": "high",
        "IT17": "high",
        "IT18": "high",
        "IT27": "medium",
    }
    rules = {
        issue_type_id: {"policy_severity_floor": floor} for issue_type_id, floor in floors.items()
    }
    predictions = apply_policy_severity_floor(
        pd.Series(list(floors)),
        pd.Series(["low"] * len(floors)),
        rules,
    )
    assert predictions.tolist() == list(floors.values())


def test_explicit_safety_override_is_followed_by_severity_floor():
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, texts):
            return pd.Series([self.value] * len(texts))

    feedback = pd.DataFrame(
        [
            {
                "feedback_id": "F1",
                "channel": "review",
                "created_at": "2026-03-30",
                "store_id": "S1",
                "rating": 1,
                "text": "Wet floor by entrance",
            }
        ]
    )
    bundle = {
        "models": {
            "issue_type_id": ConstantModel("NI02"),
            "severity": ConstantModel("low"),
            "actionable": ConstantModel(0),
        },
        "taxonomy": {
            "IT08": {"policy_severity_floor": "high"},
            "NI02": {"policy_severity_floor": "low"},
        },
    }

    predictions = predict_records(feedback, bundle)

    assert predictions["issue_type_id"].tolist() == ["IT08"]
    assert predictions["severity"].tolist() == ["high"]
    assert predictions["actionable"].tolist() == [0]


@pytest.mark.parametrize(
    ("text", "expected_issue"),
    [
        ("Wet floor by entrance; no warning sign", "IT08"),
        ("Loose wire near billing", "IT08"),
        ("Cockroach near bakery", "IT06"),
        ("Grocery bins ke paas keede", "IT06"),
        ("FRZR 4 not cooling since morning", "IT10"),
        ("Freezer warm; frozen items thawing", "IT10"),
        ("Sesame present but not declared on label", "IT18"),
        ("Egg product marked veg", "IT18"),
        ("Bread expired yesterday", "IT16"),
        ("Hair in dal", "IT17"),
        ("Toor dal madhe kes hote", "IT17"),
        ("Cashier shouted at me", "IT04"),
        ("Guard ne sabke saamne mujhe chor bola", "IT04"),
    ],
)
def test_explicit_policy_rules_override_for_clear_evidence(text, expected_issue):
    predictions = apply_explicit_policy_rules(
        pd.Series([text]),
        pd.Series(["NI02"]),
    )
    assert predictions.tolist() == [expected_issue]


@pytest.mark.parametrize(
    "text",
    [
        "Staff helped me find flour",
        "Staff did not shout at me",
        "The label looks wrong",
        "The label lists sesame",
        "The ingredient label is not wrong",
        "The floor is not wet",
        "The freezer is not warm",
        "The store looks dirty",
        "Fresh bread was stale",
    ],
)
def test_explicit_policy_rules_leave_ambiguous_or_negative_text_unchanged(text):
    predictions = apply_explicit_policy_rules(
        pd.Series([text]),
        pd.Series(["NI02"]),
    )
    assert predictions.tolist() == ["NI02"]
