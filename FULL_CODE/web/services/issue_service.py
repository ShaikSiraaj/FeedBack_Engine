"""Merge detector and weekly trend flags with persistent triage statuses."""

from services.data_service import get_emerging_issues
from services.report_service import build_weekly_report, get_report_options
from services.storage_service import load_issue_statuses, update_issue_status

VALID_STATUSES = {"new", "investigating", "archived"}


def get_managed_issues(week_index, region):
    """Return one region/week's emerging issues with stable keys and saved status."""
    options = get_report_options()
    selected_week = next(
        (week for week in options["weeks"] if week["week_index"] == int(week_index)),
        None,
    )
    if selected_week is None:
        raise ValueError(f"Reporting week {week_index} is not in the project calendar.")
    if region not in options["regions"]:
        raise ValueError(f"Unknown reporting region: {region}")

    flags = []
    detector_flags = get_emerging_issues()
    if not detector_flags.empty:
        week_flags = detector_flags.reset_index(drop=True).reset_index(names="flag_index")
        week_flags = week_flags[
            week_flags["as_of_week_start"].astype(str).eq(selected_week["week_start"])
        ]
        if not week_flags.empty:
            for row in week_flags.to_dict(orient="records"):
                issue_region = str(row.get("region", ""))
                if issue_region.casefold() != region.casefold():
                    continue
                flags.append(
                    {
                        "issue_key": (
                            f"{selected_week['week_start']}:{row['store_id']}:"
                            f"{row['issue_type_id']}"
                        ),
                        "week_index": int(week_index),
                        "week_start": selected_week["week_start"],
                        "store_id": str(row["store_id"]),
                        "store_name": row.get("store_name") or row["store_id"],
                        "issue_type_id": str(row["issue_type_id"]),
                        "issue_type": row.get("issue_type") or row["issue_type_id"],
                        "description": row.get("description") or "",
                        "owner_function": row.get("owner_function") or "Unassigned",
                        "policy_severity_floor": row.get("policy_severity_floor") or "low",
                        "confidence": float(row.get("confidence", 0)),
                        "evidence_count": int(row.get("evidence_count", 0)),
                        "evidence_feedback_ids": row.get("evidence_feedback_ids", ""),
                        "source": "Detector",
                        "flag_index": int(row["flag_index"]),
                    }
                )

    report = build_weekly_report(week_index, region)
    existing_keys = {flag["issue_key"] for flag in flags}
    for spike in report["emerging_flags"]:
        key = f"{selected_week['week_start']}:{spike['store_id']}:{spike['issue_type_id']}"
        if key in existing_keys:
            continue
        flags.append(
            {
                "issue_key": key,
                "week_index": int(week_index),
                "week_start": selected_week["week_start"],
                "store_id": spike["store_id"],
                "store_name": spike["store_name"],
                "issue_type_id": spike["issue_type_id"],
                "issue_type": spike["issue_type"],
                "description": (
                    f"{spike['mentions']} mentions this week versus an "
                    f"{spike['baseline']:.1f}-mention eight-week average."
                ),
                "owner_function": spike["owner_function"],
                "policy_severity_floor": spike["policy_severity_floor"],
                "confidence": min(1.0, spike["z_score"] / 10),
                "evidence_count": int(spike["mentions"]),
                "evidence_feedback_ids": spike["evidence_feedback_ids"],
                "flag_index": None,
                "source": "Weekly trend",
            }
        )

    saved_statuses = load_issue_statuses()
    for flag in flags:
        saved = saved_statuses.get(flag["issue_key"])
        flag["status"] = saved["status"] if saved else "new"
        flag["updated_at"] = saved["updated_at"] if saved else None
    return sorted(
        flags,
        key=lambda flag: (
            {"new": 0, "investigating": 1, "archived": 2}[flag["status"]],
            -float(flag["confidence"]),
            -int(flag["evidence_count"]),
        ),
    )


def set_managed_issue_status(issue_key, status, week_index, region):
    """Check that the issue belongs to the selected report before persisting status."""
    if status not in VALID_STATUSES:
        raise ValueError("Status must be new, investigating, or archived.")
    available = get_managed_issues(week_index, region)
    if issue_key not in {flag["issue_key"] for flag in available}:
        raise ValueError("This issue is not available in the selected week and region.")
    update_issue_status(issue_key, status)
    return status
