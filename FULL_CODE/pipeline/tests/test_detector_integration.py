import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

FULL_CODE = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detector():
    return _load_module("r5_detector", FULL_CODE / "detect.py")


def _official_evaluator():
    return _load_module("r5_official_evaluator", FULL_CODE / "evalutor" / "evalute.py")


def _fixture_data():
    feedback = pd.DataFrame(
        [
            {
                "feedback_id": "F1",
                "created_at": "2026-04-01 10:00:00",
                "store_id": "S1",
                "text": "Queue was long",
            },
            {
                "feedback_id": "F2",
                "created_at": "2026-04-05 10:00:00",
                "store_id": "S1",
                "text": "Queue was long again",
            },
            {
                "feedback_id": "F3",
                "created_at": "2026-04-06 00:00:00",
                "store_id": "S1",
                "text": "Queue was long next week",
            },
        ]
    )
    predictions = pd.DataFrame(
        {
            "feedback_id": ["F1", "F2", "F3"],
            "issue_type_id": ["IT01", "IT01", "IT01"],
            "severity": ["medium", "medium", "medium"],
            "actionable": [1, 1, 1],
        }
    )
    stores = pd.DataFrame(
        [{"store_id": "S1", "store_name": "Saffron Market Central", "avg_weekly_footfall": 100}]
    )
    taxonomy = pd.DataFrame(
        [{"issue_type_id": "IT01", "policy_severity_floor": "low", "policy_reason": ""}]
    )
    calendar = pd.DataFrame(
        [
            {"week_start": "2026-03-30", "week_end": "2026-04-05"},
            {"week_start": "2026-04-06", "week_end": "2026-04-12"},
        ]
    )
    return feedback, predictions, stores, taxonomy, calendar


def test_detector_evidence_is_semicolon_separated_and_officially_valid(tmp_path):
    detector = _detector()
    evaluator = _official_evaluator()
    feedback, predictions, stores, taxonomy, calendar = _fixture_data()

    flags = detector.detect_flags(
        feedback, predictions, stores, taxonomy, calendar, "2026-04-01"
    )

    assert flags.columns.tolist() == evaluator.FLAG_COLS
    assert len(flags) == 1
    assert pd.to_numeric(flags["confidence"]).between(0, 1).all()
    ids = flags.iloc[0]["evidence_feedback_ids"].split(";")
    assert set(ids) == {"F1", "F2"}
    assert "F3" not in ids
    assert "|" not in flags.iloc[0]["evidence_feedback_ids"]
    assert flags.iloc[0]["as_of_week_start"] == "2026-03-30"

    feedback_path = tmp_path / "feedback.csv"
    stores_path = tmp_path / "store_master.csv"
    flags_path = tmp_path / "flags.csv"
    feedback.to_csv(feedback_path, index=False)
    stores.to_csv(stores_path, index=False)
    flags.to_csv(flags_path, index=False)
    flags = evaluator.load_flags(flags_path)
    evidence = evaluator.Evidence(feedback_path, stores_path)
    prediction_types = dict(zip(predictions["feedback_id"], predictions["issue_type_id"]))
    week_indexes = {week: index for index, week in enumerate(evaluator.WEEK_STARTS, start=1)}
    valid, _ = evaluator.check_evidence(
        flags, prediction_types, evidence, week_indexes
    )
    assert valid.tolist() == [True]

    for column, value in (
        ("store_id", "S2"),
        ("issue_type_id", "IT02"),
        ("as_of_week_start", "2026-04-06"),
    ):
        invalid_flags = flags.copy()
        invalid_flags.loc[0, column] = value
        invalid, _ = evaluator.check_evidence(
            invalid_flags, prediction_types, evidence, week_indexes
        )
        assert invalid.tolist() == [False]


def test_detector_requires_two_evidence_ids_and_limits_weekly_flags():
    detector = _detector()
    feedback, predictions, stores, taxonomy, calendar = _fixture_data()
    one_record = feedback.iloc[[0]]
    one_prediction = predictions.iloc[[0]]
    assert detector.detect_flags(
        one_record, one_prediction, stores, taxonomy, calendar, "2026-04-01"
    ).empty

    rows = []
    predicted_rows = []
    stores_list = []
    taxonomy_list = []
    for store_number in range(5):
        store_id = f"S{store_number}"
        issue_type_id = f"IT{store_number + 1:02d}"
        stores_list.append(
            {
                "store_id": store_id,
                "store_name": f"Saffron Market {store_id}",
                "avg_weekly_footfall": 100,
            }
        )
        taxonomy_list.append(
            {
                "issue_type_id": issue_type_id,
                "policy_severity_floor": "low",
                "policy_reason": "",
            }
        )
        for record_number in range(2):
            feedback_id = f"{store_id}-F{record_number}"
            rows.append(
                {
                    "feedback_id": feedback_id,
                    "created_at": "2026-04-02",
                    "store_id": store_id,
                    "text": "Feedback",
                }
            )
            predicted_rows.append(
                {
                    "feedback_id": feedback_id,
                    "issue_type_id": issue_type_id,
                    "severity": "medium",
                    "actionable": 1,
                }
            )

    limited = detector.detect_flags(
        pd.DataFrame(rows),
        pd.DataFrame(predicted_rows),
        pd.DataFrame(stores_list),
        pd.DataFrame(taxonomy_list),
        calendar,
        "2026-04-01",
    )
    assert len(limited) == detector.MAX_FLAGS_PER_WEEK
    assert limited.groupby(
        ["as_of_week_start", "store_id", "issue_type_id"]
    ).size().max() == 1


def test_detector_command_generator_matches_actual_cli(tmp_path):
    command_path = tmp_path / "detector_cmd.txt"
    generator = FULL_CODE / "write_detector_cmd.py"
    generated = subprocess.run(
        [sys.executable, str(generator), "--out", str(command_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert generated.stdout == ""
    command = command_path.read_text(encoding="utf-8").strip()
    assert command == (
        "uv run --project FULL_CODE/pipeline python FULL_CODE/detect.py "
        "--feedback {feedback} --as-of {asof} --out {out}"
    )
    assert command in (FULL_CODE.parent / "README.md").read_text(encoding="utf-8")

    help_result = subprocess.run(
        [sys.executable, str(FULL_CODE / "detect.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--feedback" in help_result.stdout
    assert "--as-of" in help_result.stdout
    assert "--out" in help_result.stdout
