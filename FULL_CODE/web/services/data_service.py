import pandas as pd

from services.storage_service import load_uploaded_feedback

from config import (
    CALENDAR_FILE,
    EMERGING_FLAGS_FILE,
    FEEDBACK_FILE,
    STORE_FILE,
    TAXONOMY_FILE,
)


def _read_csv(path, required=True):
    """Read a CSV safely and report a useful error if it is missing."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required data file not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")


def load_feedback():
    """Load original feedback records."""
    df = _read_csv(FEEDBACK_FILE)

    required_columns = {
        "feedback_id",
        "channel",
        "created_at",
        "store_id",
        "rating",
        "text",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"feedback.csv is missing required columns: {sorted(missing)}"
        )

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["store_id"] = df["store_id"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    uploaded = load_uploaded_feedback()
    if not uploaded.empty:
        df = pd.concat([df, uploaded], ignore_index=True)

    return df


def load_taxonomy():
    """Load issue categories and their policy metadata."""
    df = _read_csv(TAXONOMY_FILE)

    required_columns = {
        "issue_type_id",
        "issue_type",
        "description",
        "owner_function",
        "policy_severity_floor",
        "policy_reason",
        "is_actionable_category",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"issue_taxonomy.csv is missing required columns: {sorted(missing)}"
        )

    return df


def load_stores():
    """Load store master information."""
    return _read_csv(STORE_FILE)


def load_calendar():
    """Load the project week calendar."""
    return _read_csv(CALENDAR_FILE)


def load_emerging_flags():
    """Load detector-generated emerging issue flags."""
    return _read_csv(EMERGING_FLAGS_FILE, required=False)


def get_overview():
    """Calculate dashboard KPIs from the actual source files."""
    feedback = load_feedback()
    stores = load_stores()
    flags = load_emerging_flags()
    uploaded = load_uploaded_feedback()

    high_risk_count = 0

    if not flags.empty:
        taxonomy = load_taxonomy()
        enriched = flags.merge(
            taxonomy[["issue_type_id", "policy_severity_floor"]],
            on="issue_type_id",
            how="left",
        )

        high_risk_count = int(
            enriched["policy_severity_floor"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["high", "critical"])
            .sum()
        )

    return {
        "total_feedback": int(len(feedback)),
        "number_of_stores": int(stores["store_id"].nunique())
        if "store_id" in stores.columns
        else 0,
        "emerging_issues": int(len(flags)),
        "uploaded_feedback": int(len(uploaded)),
        "high_risk_alerts": high_risk_count,
    }


def get_feedback_records(search="", channel="", store_id=""):
    """
    Return real feedback records with store names.

    Predicted issue, severity and actionability are intentionally not
    fabricated here. They will be joined from the classifier output
    once the prediction interface is connected.
    """
    feedback = load_feedback()
    stores = load_stores()

    if search:
        feedback = feedback[
            feedback["text"].str.contains(search, case=False, na=False)
            | feedback["feedback_id"].str.contains(
                search, case=False, na=False
            )
        ]

    if channel:
        feedback = feedback[feedback["channel"] == channel]

    if store_id:
        feedback = feedback[feedback["store_id"] == store_id]

    if "store_name" in stores.columns:
        feedback = feedback.merge(
            stores[["store_id", "store_name", "region"]],
            on="store_id",
            how="left",
        )

    feedback = feedback.sort_values(
        "created_at", ascending=False, na_position="last"
    )

    return feedback


def get_emerging_issues():
    """Enrich detector flags with taxonomy and store information."""
    flags = load_emerging_flags()

    if flags.empty:
        return flags

    taxonomy = load_taxonomy()
    stores = load_stores()

    flags = flags.merge(
        taxonomy[
            [
                "issue_type_id",
                "issue_type",
                "description",
                "owner_function",
                "policy_severity_floor",
                "policy_reason",
            ]
        ],
        on="issue_type_id",
        how="left",
    )

    store_columns = [
        column
        for column in ["store_id", "store_name", "city", "region"]
        if column in stores.columns
    ]

    flags = flags.merge(
        stores[store_columns],
        on="store_id",
        how="left",
    )

    flags["evidence_feedback_ids"] = (
        flags["evidence_feedback_ids"].fillna("").astype(str)
    )

    flags["evidence_count"] = flags["evidence_feedback_ids"].apply(
        lambda value: len(
            [item for item in value.split("|") if item.strip()]
        )
    )

    return flags.sort_values(
        ["confidence", "evidence_count"],
        ascending=[False, False],
    )


def get_issue_details(flag_index):
    """Return one emerging flag and its supporting original feedback."""
    flags = get_emerging_issues()

    if flag_index < 0 or flag_index >= len(flags):
        return None

    flag = flags.iloc[flag_index].to_dict()

    evidence_ids = [
        item.strip()
        for item in str(flag.get("evidence_feedback_ids", "")).split("|")
        if item.strip()
    ]

    feedback = load_feedback()
    evidence = feedback[
        feedback["feedback_id"].isin(evidence_ids)
    ].copy()

    stores = load_stores()
    if "store_name" in stores.columns:
        evidence = evidence.merge(
            stores[["store_id", "store_name"]],
            on="store_id",
            how="left",
        )

    # Keep evidence in the same order as the detector's ID list.
    evidence["_evidence_order"] = evidence["feedback_id"].map(
        {feedback_id: index for index, feedback_id in enumerate(evidence_ids)}
    )
    evidence = evidence.sort_values("_evidence_order").drop(
        columns=["_evidence_order"]
    )

    return {
        "flag": flag,
        "evidence": evidence.to_dict(orient="records"),
    }