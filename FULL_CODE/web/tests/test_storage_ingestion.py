import pandas as pd
import pytest

from services import ingestion_service, storage_service


def test_feedback_upload_validation_and_duplicate_preview(monkeypatch):
    feedback = pd.DataFrame(
        [
            {
                "feedback_id": "FB-EXISTING",
                "store_id": "ST-1",
            }
        ]
    )
    uploaded = pd.DataFrame(columns=["feedback_id"])
    stores = pd.DataFrame([{"store_id": "ST-1"}])
    monkeypatch.setattr(ingestion_service, "load_feedback", lambda: feedback)
    monkeypatch.setattr(ingestion_service, "load_uploaded_feedback", lambda: uploaded)
    monkeypatch.setattr(ingestion_service, "load_stores", lambda: stores)

    csv = (
        b"feedback_id,channel,created_at,store_id,rating,text\n"
        b"FB-EXISTING,in_app,2026-10-28,ST-1,4,Repeat\n"
        b"FB-NEW,online_review,2026-10-29,ST-1,,New feedback\n"
    )
    preview = ingestion_service.preview_feedback_upload(csv)

    assert preview["filename_records"] == 2
    assert preview["records_to_add"] == 1
    assert preview["duplicates_skipped"] == 1
    assert preview["sample"][0]["feedback_id"] == "FB-EXISTING"


def test_feedback_upload_rejects_unknown_stores(monkeypatch):
    monkeypatch.setattr(
        ingestion_service,
        "load_stores",
        lambda: pd.DataFrame([{"store_id": "ST-KNOWN"}]),
    )
    csv = (
        b"feedback_id,channel,created_at,store_id,rating,text\n"
        b"FB-1,in_app,2026-10-29,ST-UNKNOWN,4,Feedback\n"
    )

    with pytest.raises(ValueError, match="Unknown store_id"):
        ingestion_service.read_feedback_upload(csv)


def test_feedback_upload_rejects_duplicate_ids_in_same_csv(monkeypatch):
    monkeypatch.setattr(
        ingestion_service,
        "load_stores",
        lambda: pd.DataFrame([{"store_id": "ST-1"}]),
    )
    csv = (
        b"feedback_id,channel,created_at,store_id,rating,text\n"
        b"FB-1,in_app,2026-10-29,ST-1,4,First\n"
        b"FB-1,in_app,2026-10-29,ST-1,3,Duplicate\n"
    )

    with pytest.raises(ValueError, match="duplicate feedback_id"):
        ingestion_service.read_feedback_upload(csv)


def test_issue_status_persists_in_sqlite(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_service, "STORAGE_FILE", tmp_path / "state.sqlite3")

    storage_service.update_issue_status("2026-10-26:ST-1:IT01", "investigating")
    statuses = storage_service.load_issue_statuses()

    assert statuses["2026-10-26:ST-1:IT01"]["status"] == "investigating"


def test_issue_status_rejects_unknown_state(monkeypatch, tmp_path):
    monkeypatch.setattr(storage_service, "STORAGE_FILE", tmp_path / "state.sqlite3")

    with pytest.raises(ValueError, match="Status must be"):
        storage_service.update_issue_status("2026-10-26:ST-1:IT01", "resolved")
