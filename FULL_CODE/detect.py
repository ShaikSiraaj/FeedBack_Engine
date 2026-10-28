import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml


REQUIRED_OUTPUT_COLUMNS = [
    "as_of_week_start",
    "store_id",
    "issue_type_id",
    "confidence",
    "evidence_feedback_ids",
]

HIGH_RISK_TERMS = (
    "allergen",
    "allergy",
    "food poisoning",
    "food safety",
    "contamination",
    "contaminated",
    "hygiene",
    "unsafe",
    "harassment",
    "misconduct",
    "abuse",
    "mold",
    "mould",
    "rotten",
)


def load_data(feedback_path):
    """Load feedback and supporting lookup tables from the same data folder."""
    data_dir = Path(feedback_path).resolve().parent
    feedback = pd.read_csv(data_dir / "feedback.csv", low_memory=False)
    stores = pd.read_csv(data_dir / "store_master.csv", low_memory=False)
    taxonomy = pd.read_csv(data_dir / "issue_taxonomy.csv", low_memory=False)
    calendar = pd.read_csv(data_dir / "week_calendar.csv", low_memory=False)
    return feedback, stores, taxonomy, calendar


def get_model_path():
    """Resolve the trained model path from pipeline/config.yaml."""
    config_path = Path(__file__).resolve().parent / "pipeline" / "config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    base_dir = config_path.parent
    models_dir = (base_dir / config["models_dir"]).resolve()
    return models_dir / "r5_classifier.joblib"


def predict_feedback(feedback, model_path):
    """Run the project's trained classifier."""
    project_src = Path(__file__).resolve().parent / "pipeline" / "src"
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

    from voice_of_store.predict import predict_records

    model_bundle = joblib.load(model_path)
    return predict_records(feedback, model_bundle)


def prepare_calendar(calendar):
    calendar = calendar.copy()
    calendar["week_start"] = pd.to_datetime(
        calendar["week_start"], errors="coerce"
    ).dt.normalize()
    calendar["week_end"] = pd.to_datetime(
        calendar["week_end"], errors="coerce"
    ).dt.normalize()
    return calendar.dropna(subset=["week_start", "week_end"]).sort_values(
        "week_start"
    )


def detect_flags(feedback, predictions, stores, taxonomy, calendar, as_of):
    """Detect candidate emerging issues for the as-of week.

    An ordinary issue must have at least two distinct reports in the current
    week and show an increase against prior weeks. High-risk issues are
    prioritized and may be flagged without a historical increase, but still
    require at least two distinct current-week reports.

    The confidence value is a transparent ranking heuristic, not a calibrated
    probability.
    """
    feedback = feedback.copy()
    predictions = predictions.copy()

    # Parse timestamps consistently and treat --as-of as an inclusive date.
    feedback["created_at"] = pd.to_datetime(
        feedback["created_at"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    cutoff = pd.Timestamp(as_of).normalize()
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    as_of_end = cutoff + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    cal = prepare_calendar(calendar)
    eligible_weeks = cal[
        (cal["week_start"] <= cutoff) & (cal["week_end"] >= cutoff)
    ]
    if eligible_weeks.empty:
        # Fall back to the latest week whose start is not after the as-of date.
        eligible_weeks = cal[cal["week_start"] <= cutoff]
    if eligible_weeks.empty:
        raise ValueError(
            f"No calendar week starts on or before {as_of}. "
            "Check the --as-of date and week_calendar.csv."
        )

    current_week = eligible_weeks.iloc[-1]
    week_start = current_week["week_start"]
    week_end = min(current_week["week_end"], cutoff)

    feedback = feedback[
        feedback["created_at"].notna()
        & (feedback["created_at"] <= as_of_end)
        & feedback["store_id"].notna()
        & feedback["feedback_id"].notna()
    ].copy()

    # Merge by stable feedback ID, not row position.
    predictions = predictions.drop_duplicates("feedback_id")
    feedback = feedback.merge(
        predictions,
        on="feedback_id",
        how="inner",
        validate="one_to_one",
    )

    # Use the supplied calendar's week boundaries for all records.
    # This supports non-Monday week starts if the calendar defines them.
    feedback["week_start"] = pd.NaT
    for row in cal.itertuples(index=False):
        mask = (
            (feedback["created_at"].dt.normalize() >= row.week_start)
            & (feedback["created_at"].dt.normalize() <= row.week_end)
        )
        feedback.loc[mask, "week_start"] = row.week_start

    # Restrict history to weeks ending before the current week, plus the
    # current week through the requested as-of date.
    feedback = feedback.dropna(subset=["week_start"])
    current = feedback[
        (feedback["week_start"] == week_start)
        & (feedback["created_at"] <= as_of_end)
        & (feedback["created_at"].dt.normalize() <= week_end)
    ].copy()

    if current.empty:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    history = feedback[feedback["week_start"] < week_start].copy()

    # Attach store footfall and taxonomy policy fields.
    taxonomy_columns = [
        col for col in
        ["issue_type_id", "policy_severity_floor", "policy_reason", "issue_type"]
        if col in taxonomy.columns
    ]
    taxonomy_small = taxonomy[taxonomy_columns].drop_duplicates("issue_type_id")
    current = current.merge(taxonomy_small, on="issue_type_id", how="left")
    history = history.merge(taxonomy_small, on="issue_type_id", how="left")

    store_columns = ["store_id"]
    if "avg_weekly_footfall" in stores.columns:
        store_columns.append("avg_weekly_footfall")
    stores_small = stores[store_columns].drop_duplicates("store_id")
    current = current.merge(stores_small, on="store_id", how="left")

    if "avg_weekly_footfall" not in current.columns:
        current["avg_weekly_footfall"] = pd.NA
    current["avg_weekly_footfall"] = pd.to_numeric(
        current["avg_weekly_footfall"], errors="coerce"
    )

    current["text"] = current["text"].fillna("").astype(str)
    current["text_lower"] = current["text"].str.lower()

    # Current counts are per store and issue type, based on unique feedback.
    current_counts = (
        current.groupby(["store_id", "issue_type_id"], dropna=True)
        .agg(
            current_count=("feedback_id", "nunique"),
            evidence_text=("text_lower", lambda s: " ".join(s)),
            predicted_severities=("severity", lambda s: list(s.dropna().astype(str))),
            evidence_ids=("feedback_id", lambda s: sorted(set(s.dropna().astype(str)))),
        )
        .reset_index()
    )

    # Build historical weekly counts, then average the prior available weeks.
    
    # Build historical weekly counts, including zero-report weeks.
    history_week_starts = (
        cal.loc[cal["week_start"] < week_start, "week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    observed_history_weeks = len(history_week_starts)

    if observed_history_weeks == 0:
        historical_counts = pd.DataFrame(
            columns=[
                "store_id", "issue_type_id",
                "baseline_count", "previous_count"
            ]
        )
    else:
        weekly = (
            history.groupby(
                ["week_start", "store_id", "issue_type_id"],
                dropna=True
            )["feedback_id"]
            .nunique()
            .rename("count")
            .reset_index()
        )

        # Build a store/issue/week grid so missing reports count as zero.
        pairs = current_counts[
            ["store_id", "issue_type_id"]
        ].drop_duplicates()

        weeks_df = pd.DataFrame({
            "week_start": history_week_starts
        })

        grid = pairs.merge(weeks_df, how="cross")
        grid = grid.merge(
            weekly,
            on=["week_start", "store_id", "issue_type_id"],
            how="left"
        )
        grid["count"] = grid["count"].fillna(0)

        baseline = (
            grid.groupby(["store_id", "issue_type_id"])["count"]
            .mean()
            .rename("baseline_count")
            .reset_index()
        )

        previous_week_start = max(history_week_starts)
        previous = (
            grid[grid["week_start"] == previous_week_start][
                ["store_id", "issue_type_id", "count"]
            ]
            .rename(columns={"count": "previous_count"})
        )

        historical_counts = baseline.merge(
            previous,
            on=["store_id", "issue_type_id"],
            how="outer"
        )

    current_counts = current_counts.merge(
        historical_counts, on=["store_id", "issue_type_id"], how="left"
    )
    current_counts["baseline_count"] = current_counts["baseline_count"].fillna(0.0)
    current_counts["previous_count"] = current_counts["previous_count"].fillna(0.0)

    footfall_lookup = current[
        ["store_id", "avg_weekly_footfall"]
    ].drop_duplicates("store_id")
    current_counts = current_counts.merge(
        footfall_lookup, on="store_id", how="left"
    )

    taxonomy_meta = taxonomy_small.copy()
    current_counts = current_counts.merge(
        taxonomy_meta, on="issue_type_id", how="left"
    )

    results = []
    for row in current_counts.itertuples(index=False):
        evidence_ids = list(row.evidence_ids)
        count = len(evidence_ids)
        if count < 2:
            continue

        policy_reason = str(getattr(row, "policy_reason", "") or "").lower()

        # Evaluate risk at the individual-record level instead of combining
        # every feedback text in the store/issue group.
        group_records = current[
            (current["store_id"] == row.store_id)
            & (current["issue_type_id"] == row.issue_type_id)
        ]

        risk_records = group_records[
            group_records["text_lower"].apply(
                lambda text: any(term in text for term in HIGH_RISK_TERMS)
            )
            | group_records["severity"].fillna("").astype(str).str.lower().isin(
                ["critical", "high"]
            )
        ]

        policy_risk = any(term in policy_reason for term in HIGH_RISK_TERMS)
        high_risk = not risk_records.empty or policy_risk

        baseline = float(row.baseline_count)
        previous_count = float(row.previous_count)
        current_count = float(row.current_count)

        # Emerging rule: current count must exceed both the historical average
        # and the most recent previous week. If there is no history, two or
        # more reports qualify as a provisional candidate.
        if observed_history_weeks > 0:
            increased = current_count > baseline and current_count > previous_count
            if not increased and not high_risk:
                continue
        else:
            increased = False

        footfall = getattr(row, "avg_weekly_footfall", pd.NA)
        try:
            footfall = float(footfall)
            footfall_rate = (current_count / footfall * 1000) if footfall > 0 else 0.0
        except (TypeError, ValueError):
            footfall_rate = 0.0

        volume_score = min(count / 5.0, 1.0)
        if baseline > 0:
            growth_score = min(max((current_count - baseline) / max(baseline, 1.0), 0.0), 1.0)
        else:
            growth_score = 1.0 if current_count >= 2 else 0.0

        confidence = 0.50 * volume_score + 0.30 * growth_score
        if high_risk:
            confidence += 0.20
        confidence = round(min(max(confidence, 0.0), 0.99), 4)

        results.append({
            "as_of_week_start": week_start.strftime("%Y-%m-%d"),
            "store_id": row.store_id,
            "issue_type_id": row.issue_type_id,
            "confidence": confidence,
            "evidence_feedback_ids": "|".join(evidence_ids),
            "_high_risk": high_risk,
            "_count": count,
            "_footfall_rate": round(footfall_rate, 4),
            "_baseline_count": round(baseline, 4),
            "_previous_count": int(previous_count),
            "_increased": increased,
        })

    if not results:
        return pd.DataFrame(columns=REQUIRED_OUTPUT_COLUMNS)

    output = pd.DataFrame(results)
    output = output.sort_values(
        ["_high_risk", "_increased", "_footfall_rate", "_count", "confidence"],
        ascending=[False, False, False, False, False],
    )
    return output[REQUIRED_OUTPUT_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description="Detect emerging store issues")
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    feedback, stores, taxonomy, calendar = load_data(args.feedback)
    print("Feedback rows:", len(feedback))
    print("Stores:", len(stores))
    print("Issue types:", len(taxonomy))
    print("Calendar rows:", len(calendar))

    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Train the model first."
        )

    predictions = predict_feedback(feedback, model_path)
    print("Predictions generated:", len(predictions))

    flags = detect_flags(
        feedback, predictions, stores, taxonomy, calendar, args.as_of
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(output_path, index=False)

    print("As-of date:", args.as_of)
    print("Emerging flags:", len(flags))
    print("Output saved to:", output_path.resolve())
    print(flags.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
