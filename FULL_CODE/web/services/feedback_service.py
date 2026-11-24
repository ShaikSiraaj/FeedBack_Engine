import pandas as pd

from services.data_service import load_stores, load_taxonomy
from services.prediction_service import get_feedback_with_predictions


def get_feedback(
    search="", channel="", store_id="", severity="", actionable="",
    issue_type_id="", region=""
):
    """Return real feedback enriched with model predictions and metadata."""
    df = get_feedback_with_predictions()
    stores = load_stores()
    taxonomy = load_taxonomy()

    if "store_name" in stores.columns:
        store_columns = [
            col for col in ["store_id", "store_name", "city", "region"]
            if col in stores.columns
        ]
        df = df.merge(stores[store_columns], on="store_id", how="left")

    taxonomy_columns = [
        "issue_type_id", "issue_type", "description", "owner_function",
        "policy_severity_floor", "policy_reason"
    ]
    df = df.merge(
        taxonomy[taxonomy_columns],
        on="issue_type_id",
        how="left",
    )

    if search:
        mask = (
            df["text"].str.contains(search, case=False, na=False)
            | df["feedback_id"].astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    if channel:
        df = df[df["channel"] == channel]

    if store_id:
        df = df[df["store_id"].astype(str) == str(store_id)]

    if region:
        df = df[df["region"].astype(str).str.casefold() == region.casefold()]

    if severity:
        df = df[df["severity"].astype(str).str.lower() == severity.lower()]

    if actionable in {"0", "1"}:
        df = df[
            pd.to_numeric(df["actionable"], errors="coerce").fillna(-1) == int(actionable)
        ]

    if issue_type_id:
        df = df[df["issue_type_id"].astype(str) == str(issue_type_id)]

    return df.sort_values(
        "created_at", ascending=False, na_position="last"
    )


def get_filter_options():
    df = get_feedback_with_predictions()
    stores = load_stores()
    taxonomy = load_taxonomy()

    return {
        "channels": sorted(df["channel"].dropna().astype(str).unique().tolist()),
        "stores": stores.sort_values("store_name").to_dict(orient="records")
        if "store_name" in stores.columns else [],
        "issues": taxonomy[["issue_type_id", "issue_type"]].to_dict(orient="records"),
        "severities": ["critical", "high", "medium", "low"],
    }
