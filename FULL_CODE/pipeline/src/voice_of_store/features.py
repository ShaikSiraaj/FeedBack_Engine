"""Dataset loading, validation, and text preparation for Voice-of-Store."""

from pathlib import Path

import pandas as pd

FEEDBACK_COLUMNS = {
    "feedback_id",
    "channel",
    "created_at",
    "store_id",
    "rating",
    "text",
}
LABEL_COLUMNS = {"feedback_id", "issue_type_id", "severity", "actionable"}
TAXONOMY_COLUMNS = {
    "issue_type_id",
    "owner_function",
    "policy_severity_floor",
    "is_actionable_category",
}
SEVERITY_ORDER = ("low", "medium", "high", "critical")
SEVERITY_RANK = {severity: rank for rank, severity in enumerate(SEVERITY_ORDER)}


class DataValidationError(ValueError):
    """Raised when an input dataset violates the R5 data contract."""


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {sorted(missing)}")


def _validate_id_column(frame: pd.DataFrame, column: str, name: str) -> None:
    if frame[column].isna().any():
        raise DataValidationError(f"{name}.{column} contains missing identifiers")
    if frame[column].duplicated().any():
        raise DataValidationError(f"{name}.{column} contains duplicate identifiers")


def validate_feedback(feedback: pd.DataFrame) -> pd.DataFrame:
    """Validate feedback structure and IDs; text, store, and rating may be missing."""
    _require_columns(feedback, FEEDBACK_COLUMNS, "feedback")
    _validate_id_column(feedback, "feedback_id", "feedback")
    return feedback


def validate_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Validate taxonomy keys and the supplied policy metadata."""
    _require_columns(taxonomy, TAXONOMY_COLUMNS, "issue_taxonomy")
    _validate_id_column(taxonomy, "issue_type_id", "issue_taxonomy")
    for column in ("owner_function", "policy_severity_floor", "is_actionable_category"):
        if taxonomy[column].isna().any():
            raise DataValidationError(f"issue_taxonomy.{column} contains missing values")
    invalid_floors = set(taxonomy["policy_severity_floor"]) - set(SEVERITY_RANK)
    if invalid_floors:
        raise DataValidationError(f"Unknown policy severity floors: {sorted(invalid_floors)}")
    invalid_actionability = set(taxonomy["is_actionable_category"]) - {0, 1}
    if invalid_actionability:
        raise DataValidationError(
            f"Unknown is_actionable_category values: {sorted(invalid_actionability)}"
        )
    return taxonomy


def validate_labels(labels: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Validate target values against the observed R5 taxonomy and label contract."""
    _require_columns(labels, LABEL_COLUMNS, "feedback_labels_train")
    _validate_id_column(labels, "feedback_id", "feedback_labels_train")
    for column in ("issue_type_id", "severity", "actionable"):
        if labels[column].isna().any():
            raise DataValidationError(f"feedback_labels_train.{column} contains missing values")

    unknown_issues = set(labels["issue_type_id"]) - set(taxonomy["issue_type_id"])
    if unknown_issues:
        raise DataValidationError(f"Unknown issue_type_id values: {sorted(unknown_issues)}")

    invalid_severities = set(labels["severity"]) - set(SEVERITY_RANK)
    if invalid_severities:
        raise DataValidationError(f"Unknown severity values: {sorted(invalid_severities)}")

    invalid_actionability = set(labels["actionable"]) - {0, 1}
    if invalid_actionability:
        raise DataValidationError(f"Unknown actionable values: {sorted(invalid_actionability)}")

    floor_by_issue = taxonomy.set_index("issue_type_id")["policy_severity_floor"]
    floors = labels["issue_type_id"].map(floor_by_issue)
    below_floor = labels["severity"].map(SEVERITY_RANK) < floors.map(SEVERITY_RANK)
    if below_floor.any():
        issue_ids = sorted(labels.loc[below_floor, "issue_type_id"].unique())
        raise DataValidationError(f"Labels fall below policy severity floors for: {issue_ids}")
    return labels


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read R5 feedback, labels, and taxonomy; return matched labelled rows and taxonomy."""
    data_dir = Path(data_dir)
    feedback = pd.read_csv(data_dir / "feedback.csv", low_memory=False)
    labels = pd.read_csv(data_dir / "feedback_labels_train.csv", low_memory=False)
    taxonomy = pd.read_csv(data_dir / "issue_taxonomy.csv", low_memory=False)

    validate_feedback(feedback)
    validate_taxonomy(taxonomy)
    validate_labels(labels, taxonomy)

    joined = labels.merge(
        feedback,
        on="feedback_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        count = int(unmatched.sum())
        raise DataValidationError(f"{count} labelled records do not match feedback records")
    joined = joined.drop(columns="_merge")
    joined["text"] = normalize_text(joined["text"])
    return joined, taxonomy


def normalize_text(text: pd.Series) -> pd.Series:
    """Represent missing text as empty text for the vectorizer."""
    return text.fillna("").astype(str)
