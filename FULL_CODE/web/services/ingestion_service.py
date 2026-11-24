"""Validate, preview, and append feedback CSV uploads."""

import json
from io import BytesIO

import pandas as pd

from config import MAX_UPLOAD_RECORDS
from services.data_service import load_feedback, load_stores
from services.prediction_service import predict_feedback_records
from services.storage_service import load_uploaded_feedback, save_uploaded_feedback

REQUIRED_COLUMNS = {
    "feedback_id",
    "channel",
    "created_at",
    "store_id",
    "rating",
    "text",
}


def read_feedback_upload(file_bytes):
    """Parse and validate an upload without storing it."""
    try:
        frame = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig", low_memory=False)
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        raise ValueError(f"Unable to read the uploaded CSV: {exc}") from exc

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The uploaded CSV contains no feedback records.")
    if len(frame) > MAX_UPLOAD_RECORDS:
        raise ValueError(
            f"The CSV contains {len(frame):,} records; the limit is "
            f"{MAX_UPLOAD_RECORDS:,} per upload."
        )
    if frame["feedback_id"].isna().any():
        raise ValueError("feedback_id cannot be empty.")
    frame["feedback_id"] = frame["feedback_id"].astype(str).str.strip()
    if frame["feedback_id"].eq("").any():
        raise ValueError("feedback_id cannot be empty.")
    if frame["feedback_id"].duplicated().any():
        raise ValueError("The CSV contains duplicate feedback_id values.")

    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    if frame["created_at"].isna().any():
        raise ValueError("created_at must contain a valid date and time for every record.")

    for column in ("channel", "store_id", "text"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame["channel"].eq("").any():
        raise ValueError("channel cannot be empty.")
    if frame["store_id"].eq("").any():
        raise ValueError("store_id cannot be empty.")
    known_stores = set(load_stores()["store_id"].dropna().astype(str))
    unknown_stores = sorted(set(frame["store_id"]) - known_stores)
    if unknown_stores:
        raise ValueError(
            f"Unknown store_id values: {unknown_stores[:10]}. "
            "Add the stores to store_master.csv before importing feedback."
        )

    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    if frame["rating"].notna().any() and not frame["rating"].dropna().between(0, 5).all():
        raise ValueError("rating must be between 0 and 5.")

    return frame


def preview_feedback_upload(file_bytes):
    """Validate an upload and report how many IDs are eligible to append."""
    frame = read_feedback_upload(file_bytes)
    existing_ids = set(load_feedback()["feedback_id"].astype(str))
    uploaded_ids = set(
        load_uploaded_feedback().get("feedback_id", pd.Series(dtype=str)).astype(str)
    )
    duplicate_ids = existing_ids | uploaded_ids
    duplicate_count = int(frame["feedback_id"].isin(duplicate_ids).sum())
    return {
        "filename_records": len(frame),
        "records_to_add": len(frame) - duplicate_count,
        "duplicates_skipped": duplicate_count,
        "sample": json.loads(
            frame.head(5).to_json(orient="records", date_format="iso")
        ),
    }


def import_feedback_upload(file_bytes, filename):
    """Predict and append only new feedback records to local SQLite."""
    frame = read_feedback_upload(file_bytes)
    existing_ids = set(load_feedback()["feedback_id"].astype(str))
    uploaded_ids = set(
        load_uploaded_feedback().get("feedback_id", pd.Series(dtype=str)).astype(str)
    )
    duplicate_ids = existing_ids | uploaded_ids
    new_records = frame.loc[~frame["feedback_id"].isin(duplicate_ids)].copy()
    if new_records.empty:
        return {"records_added": 0, "duplicates_skipped": len(frame)}

    predictions = predict_feedback_records(new_records)
    enriched = new_records.merge(
        predictions,
        on="feedback_id",
        how="left",
        validate="one_to_one",
    )
    added = save_uploaded_feedback(enriched.to_dict(orient="records"), filename)
    return {
        "records_added": added,
        "duplicates_skipped": len(frame) - len(new_records),
    }
