from io import BytesIO

import re
from pathlib import Path

import pandas as pd
import pytest

from services import report_service


def _report_inputs():
    weeks = pd.date_range("2026-01-05", periods=10, freq="7D")
    calendar = pd.DataFrame(
        {
            "week_index": range(1, 11),
            "week_start": weeks,
            "week_end": weeks + pd.Timedelta(days=6),
        }
    )
    records = []
    for week_index, week_start in enumerate(weeks, start=1):
        count = 4 if week_index == 10 else 1
        for offset in range(count):
            records.append(
                {
                    "feedback_id": f"FB-{week_index}-{offset}",
                    "created_at": week_start + pd.Timedelta(days=offset),
                    "store_id": "S-1",
                    "issue_type_id": "IT01",
                    "severity": "high",
                    "actionable": 1,
                }
            )
    records.append(
        {
            "feedback_id": "FB-10-other",
            "created_at": weeks[-1] + pd.Timedelta(days=1),
            "store_id": "S-1",
            "issue_type_id": "NI01",
            "severity": "low",
            "actionable": 0,
        }
    )
    feedback = pd.DataFrame(records)
    stores = pd.DataFrame(
        [{"store_id": "S-1", "store_name": "Central Store", "region": "Central"}]
    )
    taxonomy = pd.DataFrame(
        [
            {
                "issue_type_id": "IT01",
                "issue_type": "Safety concern",
                "owner_function": "Store Ops",
                "policy_severity_floor": "high",
            },
            {
                "issue_type_id": "NI01",
                "issue_type": "General feedback",
                "owner_function": "Customer Care",
                "policy_severity_floor": "low",
            },
        ]
    )
    return calendar, feedback, stores, taxonomy


def _mock_report_inputs(monkeypatch):
    calendar, feedback, stores, taxonomy = _report_inputs()
    monkeypatch.setattr(report_service, "_read_calendar", lambda: calendar)
    monkeypatch.setattr(report_service, "get_feedback_with_predictions", lambda: feedback.copy())
    monkeypatch.setattr(report_service, "load_stores", lambda: stores.copy())
    monkeypatch.setattr(report_service, "load_taxonomy", lambda: taxonomy.copy())
    return feedback


def test_weekly_report_uses_actual_week_predictions_and_prior_baseline(monkeypatch):
    _mock_report_inputs(monkeypatch)

    report = report_service.build_weekly_report(10, "Central")

    assert report["mentions"] == 5
    assert report["actionable"] == 4
    assert report["high_severity"] == 4
    assert report["stores"] == 1
    assert report["baseline"]["mentions"] == 1
    assert report["top_issues"][0]["issue_type"] == "Safety concern"
    assert report["top_issues"][0]["mentions"] == 4
    assert report["policy_watch"][0]["mentions"] == 4
    assert report["emerging_flags"][0]["store_name"] == "Central Store"
    assert report["emerging_flags"][0]["lift"] == 4


def test_weekly_report_rejects_unknown_week_and_region(monkeypatch):
    _mock_report_inputs(monkeypatch)

    with pytest.raises(ValueError, match="not in the project calendar"):
        report_service.build_weekly_report(11, "Central")
    with pytest.raises(ValueError, match="Unknown reporting region"):
        report_service.build_weekly_report(10, "West")


def test_weekly_pdf_is_one_page_and_contains_report_sections(monkeypatch):
    _mock_report_inputs(monkeypatch)
    report = report_service.build_weekly_report(10, "Central")

    pdf = report_service.generate_weekly_pdf(report)
    content = pdf.getvalue()

    assert content.startswith(b"%PDF-1.4")
    assert len(re.findall(rb"/Type\s*/Page\b", content)) == 1


def test_report_calendar_extends_for_new_uploaded_feedback(monkeypatch, tmp_path):
    calendar, _, _, _ = _report_inputs()
    calendar_path = tmp_path / "week_calendar.csv"
    calendar.drop(columns=["week_end"]).assign(
        week_end=calendar["week_end"].dt.strftime("%Y-%m-%d"),
        week_start=calendar["week_start"].dt.strftime("%Y-%m-%d"),
    ).to_csv(calendar_path, index=False)
    uploaded = pd.DataFrame(
        [{"created_at": pd.Timestamp("2026-03-16")}]
    )
    monkeypatch.setattr(report_service, "CALENDAR_FILE", Path(calendar_path))
    monkeypatch.setattr(report_service, "load_uploaded_feedback", lambda: uploaded)

    weeks = report_service.get_report_options()["weeks"]

    assert weeks[-1]["week_index"] == 11
    assert weeks[-1]["week_start"] == "2026-03-16"
