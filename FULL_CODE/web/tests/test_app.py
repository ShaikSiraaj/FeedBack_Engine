from io import BytesIO

import pandas as pd

import app as app_module


def _feedback_frame():
    return pd.DataFrame(
        [
            {
                "feedback_id": "FB-1",
                "text": "Feedback text",
                "created_at": pd.Timestamp("2026-10-30"),
                "store_id": "S-1",
                "store_name": "Central",
                "region": "West",
                "issue_type": "Delivery delay",
                "issue_type_id": "IT01",
                "owner_function": "Store Ops",
                "severity": "high",
                "actionable": 1,
                "channel": "online_review",
                "rating": 2,
            },
            {
                "feedback_id": "FB-2",
                "text": None,
                "created_at": pd.NaT,
                "store_id": "S-2",
                "store_name": "North",
                "region": None,
                "issue_type": "Other",
                "issue_type_id": "NI01",
                "owner_function": None,
                "severity": "low",
                "actionable": 0,
                "channel": "app",
                "rating": None,
            },
        ]
    )


def test_feedback_api_returns_paginated_json(monkeypatch):
    observed_filters = {}

    def fake_get_feedback(**filters):
        observed_filters.update(filters)
        return _feedback_frame()

    monkeypatch.setattr(app_module, "get_feedback", fake_get_feedback)
    response = app_module.app.test_client().get(
        "/api/feedback?severity=high&page=2&per_page=1"
    )

    assert response.status_code == 200
    assert observed_filters["severity"] == "high"
    assert response.json["total"] == 2
    assert response.json["page"] == 2
    assert response.json["records"][0]["feedback_id"] == "FB-2"
    assert response.json["records"][0]["text"] is None


def test_feedback_api_rejects_invalid_pagination():
    response = app_module.app.test_client().get("/api/feedback?page=0")

    assert response.status_code == 400
    assert "error" in response.json


def test_feedback_page_renders_records_and_filters(monkeypatch):
    monkeypatch.setattr(app_module, "get_feedback", lambda **filters: _feedback_frame())
    monkeypatch.setattr(
        app_module,
        "get_filter_options",
        lambda: {
            "channels": ["app", "online_review"],
            "stores": [],
            "issues": [],
            "severities": ["critical", "high", "medium", "low"],
        },
    )

    response = app_module.app.test_client().get("/feedback?per_page=1")

    assert response.status_code == 200
    assert b"FB-1" in response.data
    assert b"ACTIONABLE" in response.data
    assert b"Yes" in response.data
    assert b"Translate" in response.data
    assert b"Page 1 of 2" in response.data


def test_translation_api_calls_local_service(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "translate_text",
        lambda text, source, target: f"{source} to {target}: {text}",
    )

    response = app_module.app.test_client().post(
        "/api/translate",
        json={
            "text": "hello",
            "source_language": "en",
            "target_language": "es",
        },
    )

    assert response.status_code == 200
    assert response.json["translation"] == "en to es: hello"
    assert response.json["engine"] == "Argos Translate (local)"


def test_translation_api_rejects_invalid_payload():
    response = app_module.app.test_client().post(
        "/api/translate",
        json={"text": "hello", "source_language": "en"},
    )

    assert response.status_code == 400
    assert "error" in response.json


def test_translation_api_reports_missing_local_model(monkeypatch):
    def no_local_model(*args):
        raise app_module.TranslationUnavailableError("Install a local model.")

    monkeypatch.setattr(app_module, "translate_text", no_local_model)

    response = app_module.app.test_client().post(
        "/api/translate",
        json={
            "text": "hello",
            "source_language": "en",
            "target_language": "es",
        },
    )

    assert response.status_code == 503
    assert response.json["error"] == "Install a local model."


def test_weekly_report_page_renders_week_selector_and_preview(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_report_options",
        lambda: {
            "weeks": [{"week_index": 1, "week_start": "2026-01-05", "week_end": "2026-01-11"}],
            "regions": ["Central"],
        },
    )
    monkeypatch.setattr(
        app_module,
        "build_weekly_report",
        lambda week, region: {
            "week_index": week,
            "week_start": "2026-01-05",
            "week_end": "2026-01-11",
            "region": region,
            "stores": 1,
            "mentions": 3,
            "actionable": 2,
            "high_severity": 1,
            "baseline": None,
            "top_issues": [],
            "policy_watch": [],
            "emerging_flags": [],
        },
    )

    response = app_module.app.test_client().get("/reports")

    assert response.status_code == 200
    assert b"Weekly Reports" in response.data
    assert b"Download PDF" in response.data
    assert b"3" in response.data


def test_weekly_report_pdf_download(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "build_weekly_report",
        lambda week, region: {
            "week_index": week,
            "week_start": "2026-01-05",
        },
    )
    monkeypatch.setattr(app_module, "generate_weekly_pdf", lambda report: BytesIO(b"%PDF-1.4"))

    response = app_module.app.test_client().get("/reports/1/Central/pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert "storepulse-central-week-1-2026.pdf" in response.headers["Content-Disposition"]


def test_feedback_api_passes_actionability_filter(monkeypatch):
    observed = {}

    def fake_get_feedback(**filters):
        observed.update(filters)
        return _feedback_frame().iloc[0:0]

    monkeypatch.setattr(app_module, "get_feedback", fake_get_feedback)

    response = app_module.app.test_client().get("/api/feedback?actionable=1")

    assert response.status_code == 200
    assert observed["actionable"] == "1"


def test_csv_preview_endpoint_returns_duplicate_summary(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "preview_feedback_upload",
        lambda content: {
            "filename_records": 3,
            "records_to_add": 2,
            "duplicates_skipped": 1,
            "sample": [],
        },
    )

    response = app_module.app.test_client().post(
        "/api/data/preview",
        data={"file": (BytesIO(b"feedback_id\nFB-1"), "feedback.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["records_to_add"] == 2
    assert response.json["duplicates_skipped"] == 1


def test_csv_import_endpoint_reports_added_records(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "import_feedback_upload",
        lambda content, filename: {"records_added": 2, "duplicates_skipped": 1},
    )

    response = app_module.app.test_client().post(
        "/api/data/upload",
        data={"file": (BytesIO(b"feedback_id\nFB-1"), "feedback.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert response.json["records_added"] == 2
