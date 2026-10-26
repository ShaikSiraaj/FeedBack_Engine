"""Inference helpers for a trained R5 classification bundle."""

from typing import Any

import pandas as pd

from voice_of_store.features import SEVERITY_RANK, normalize_text, validate_feedback


def apply_policy_severity_floor(
    issue_type_ids: pd.Series,
    severities: pd.Series,
    taxonomy: dict[str, dict[str, Any]],
) -> pd.Series:
    """Raise predicted severity to the taxonomy floor without lowering it."""
    adjusted = []
    for issue_type_id, severity in zip(issue_type_ids, severities):
        if issue_type_id not in taxonomy:
            raise ValueError(f"Unknown predicted issue_type_id: {issue_type_id}")
        if severity not in SEVERITY_RANK:
            raise ValueError(f"Unknown predicted severity: {severity}")
        floor = taxonomy[issue_type_id]["policy_severity_floor"]
        if floor not in SEVERITY_RANK:
            raise ValueError(f"Unknown policy severity floor for {issue_type_id}: {floor}")
        adjusted.append(max((severity, floor), key=SEVERITY_RANK.__getitem__))
    return pd.Series(adjusted, index=severities.index, name=severities.name)


def predict_records(
    feedback: pd.DataFrame, model_bundle: dict[str, Any]
) -> pd.DataFrame:
    """Predict the three targets and enforce taxonomy severity floors."""
    validate_feedback(feedback)
    models = model_bundle["models"]
    text = normalize_text(feedback["text"])
    predictions = pd.DataFrame(
        {
            "feedback_id": feedback["feedback_id"].to_numpy(),
            "issue_type_id": models["issue_type_id"].predict(text),
            "severity": models["severity"].predict(text),
            "actionable": models["actionable"].predict(text).astype(int),
        }
    )
    predictions["severity"] = apply_policy_severity_floor(
        predictions["issue_type_id"],
        predictions["severity"],
        model_bundle["taxonomy"],
    )
    return predictions
