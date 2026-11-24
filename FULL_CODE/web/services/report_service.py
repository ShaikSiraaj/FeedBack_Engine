"""Build data-backed weekly regional briefs and PDF downloads."""

from datetime import date
from io import BytesIO
from math import sqrt
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from config import CALENDAR_FILE
from services.data_service import load_stores, load_taxonomy
from services.prediction_service import get_feedback_with_predictions
from services.storage_service import load_uploaded_feedback

SEVERITY_WEIGHT = {"low": 1.0, "medium": 1.35, "high": 1.8, "critical": 2.4}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
POLICY_SEVERITIES = {"high", "critical"}
INK = colors.HexColor("#20283A")
MUTED = colors.HexColor("#687386")
PURPLE = colors.HexColor("#6258E8")
LIGHT_PURPLE = colors.HexColor("#F0EFFF")
LINE = colors.HexColor("#E5E8EF")
PALE = colors.HexColor("#F7F8FB")
RED = colors.HexColor("#B7384D")
ORANGE = colors.HexColor("#B95D35")
GREEN = colors.HexColor("#268362")
SEVERITY_COLORS = {
    "critical": RED,
    "high": ORANGE,
    "medium": colors.HexColor("#A87919"),
    "low": GREEN,
}


def _read_calendar():
    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(f"Required week calendar not found: {CALENDAR_FILE}")
    calendar = pd.read_csv(CALENDAR_FILE)
    required = {"week_index", "week_start", "week_end"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"week_calendar.csv is missing columns: {sorted(missing)}")
    calendar["week_start"] = pd.to_datetime(calendar["week_start"], errors="coerce")
    calendar["week_end"] = pd.to_datetime(calendar["week_end"], errors="coerce")
    calendar["week_index"] = pd.to_numeric(calendar["week_index"], errors="coerce")
    calendar = calendar.dropna(subset=["week_index", "week_start", "week_end"])
    calendar["week_index"] = calendar["week_index"].astype(int)
    if calendar.empty:
        raise ValueError("week_calendar.csv contains no valid reporting weeks.")
    calendar = calendar.sort_values("week_start").reset_index(drop=True)

    uploaded = load_uploaded_feedback()
    if not uploaded.empty:
        latest_upload = pd.to_datetime(uploaded["created_at"], errors="coerce").max()
        last_week = calendar.iloc[-1]
        next_start = last_week["week_end"] + pd.Timedelta(days=1)
        next_index = int(last_week["week_index"]) + 1
        while next_start <= latest_upload:
            if next_index > int(last_week["week_index"]) + 260:
                raise ValueError(
                    "Uploaded feedback extends more than five years beyond the "
                    "reporting calendar. Update week_calendar.csv before reporting."
                )
            calendar.loc[len(calendar)] = {
                "week_index": next_index,
                "week_start": next_start,
                "week_end": next_start + pd.Timedelta(days=6),
            }
            next_start += pd.Timedelta(days=7)
            next_index += 1

    return calendar.sort_values("week_start").reset_index(drop=True)


def get_report_options():
    """Return only weeks and regions present in the supplied source data."""
    calendar = _read_calendar()
    stores = load_stores()
    if "region" not in stores.columns:
        raise ValueError("store_master.csv is missing the required region column.")
    regions = sorted(stores["region"].dropna().astype(str).unique().tolist())
    return {
        "weeks": [
            {
                "week_index": int(row.week_index),
                "week_start": row.week_start.strftime("%Y-%m-%d"),
                "week_end": row.week_end.strftime("%Y-%m-%d"),
            }
            for row in calendar.itertuples(index=False)
        ],
        "regions": regions,
    }


def _load_report_feedback():
    """Join original records, model predictions, stores and issue metadata."""
    feedback = get_feedback_with_predictions()
    stores = load_stores()
    taxonomy = load_taxonomy()

    store_columns = [
        column
        for column in ("store_id", "store_name", "city", "region")
        if column in stores.columns
    ]
    taxonomy_columns = [
        column
        for column in (
            "issue_type_id",
            "issue_type",
            "owner_function",
            "policy_severity_floor",
        )
        if column in taxonomy.columns
    ]
    if "store_id" not in store_columns or "region" not in store_columns:
        raise ValueError("store_master.csv must include store_id and region columns.")
    if not {"issue_type_id", "issue_type", "owner_function", "policy_severity_floor"} <= set(
        taxonomy_columns
    ):
        raise ValueError("issue_taxonomy.csv is missing weekly report metadata.")

    return (
        feedback.merge(stores[store_columns], on="store_id", how="left", validate="many_to_one")
        .merge(
            taxonomy[taxonomy_columns],
            on="issue_type_id",
            how="left",
            validate="many_to_one",
        )
    )


def _period_feedback(feedback, start, end, region):
    dates = pd.to_datetime(feedback["created_at"], errors="coerce")
    mask = dates.ge(start) & dates.lt(end + pd.Timedelta(days=1))
    mask &= feedback["region"].astype(str).str.casefold().eq(region.casefold())
    return feedback.loc[mask].copy()


def _issue_counts(frame):
    if frame.empty:
        return pd.DataFrame(
            columns=["issue_type_id", "mentions", "actionable", "high_severity", "stores"]
        )
    frame = frame.copy()
    frame["actionable"] = pd.to_numeric(frame["actionable"], errors="coerce").fillna(0)
    frame["severity"] = frame["severity"].fillna("low").astype(str).str.lower()
    return (
        frame.groupby("issue_type_id", dropna=False)
        .agg(
            mentions=("feedback_id", "count"),
            actionable=("actionable", "sum"),
            high_severity=("severity", lambda values: values.isin(["high", "critical"]).sum()),
            stores=("store_id", "nunique"),
        )
        .reset_index()
    )


def _emerging_spikes(feedback, calendar, week_row, region, week_data):
    """Compare this week's store/category counts with the prior eight calendar weeks."""
    previous_weeks = calendar[calendar["week_start"] < week_row.week_start].tail(8)
    if len(previous_weeks) < 4 or week_data.empty:
        return []

    history = feedback.loc[
        feedback["region"].astype(str).str.casefold().eq(region.casefold())
    ].copy()
    history["created_at"] = pd.to_datetime(history["created_at"], errors="coerce")
    historical_counts = []
    for row in previous_weeks.itertuples(index=False):
        counts = _issue_counts(_period_feedback(history, row.week_start, row.week_end, region))
        if counts.empty:
            continue
        counts["week_start"] = row.week_start
        historical_counts.append(counts[["issue_type_id", "mentions", "week_start"]])

    if historical_counts:
        history_counts = pd.concat(historical_counts, ignore_index=True)
        baseline = (
            history_counts.groupby("issue_type_id")["mentions"]
            .agg(["mean", "std"])
            .to_dict(orient="index")
        )
    else:
        baseline = {}

    store_history = []
    for row in previous_weeks.itertuples(index=False):
        prior = _period_feedback(history, row.week_start, row.week_end, region)
        counts = (
            prior.groupby(["store_id", "issue_type_id"])
            .size()
            .rename("mentions")
            .reset_index()
        )
        counts["week_start"] = row.week_start
        store_history.append(counts)
    store_history_frame = (
        pd.concat(store_history, ignore_index=True)
        if store_history
        else pd.DataFrame(columns=["store_id", "issue_type_id", "mentions", "week_start"])
    )

    flag_rows = []
    current_counts = (
        week_data.groupby(["store_id", "issue_type_id"])
        .size()
        .rename("mentions")
        .reset_index()
    )
    issue_names = week_data.drop_duplicates("issue_type_id").set_index("issue_type_id")
    store_names = week_data.drop_duplicates("store_id").set_index("store_id")
    for row in current_counts.itertuples(index=False):
        if row.mentions < 3:
            continue
        baseline_rows = store_history_frame[
            (store_history_frame["store_id"] == row.store_id)
            & (store_history_frame["issue_type_id"] == row.issue_type_id)
        ]
        weekly_history = baseline_rows.set_index("week_start")["mentions"].reindex(
            previous_weeks["week_start"], fill_value=0
        )
        mean = float(weekly_history.mean())
        if mean <= 0:
            continue
        deviation = float(weekly_history.std(ddof=0))
        standard_error = deviation if deviation > 0 else sqrt(mean)
        z_score = (row.mentions - mean) / standard_error
        lift = row.mentions / mean
        if z_score < 3 or lift < 2:
            continue

        store_name = store_names.loc[row.store_id].get("store_name")
        issue_type = issue_names.loc[row.issue_type_id].get("issue_type")
        flag_rows.append(
            {
                "store_name": str(
                    row.store_id if pd.isna(store_name) or not store_name else store_name
                ),
                "store_id": str(row.store_id),
                "issue_type_id": str(row.issue_type_id),
                "issue_type": str(
                    row.issue_type_id if pd.isna(issue_type) or not issue_type else issue_type
                ),
                "policy_severity_floor": str(
                    issue_names.loc[row.issue_type_id].get("policy_severity_floor", "low")
                ),
                "owner_function": str(
                    issue_names.loc[row.issue_type_id].get("owner_function", "Unassigned")
                ),
                "mentions": int(row.mentions),
                "evidence_feedback_ids": "|".join(
                    week_data.loc[
                        (week_data["store_id"].astype(str) == str(row.store_id))
                        & (
                            week_data["issue_type_id"].astype(str)
                            == str(row.issue_type_id)
                        ),
                        "feedback_id",
                    ].astype(str).tolist()
                ),
                "baseline": mean,
                "z_score": z_score,
                "lift": lift,
            }
        )

    return sorted(flag_rows, key=lambda item: (-item["z_score"], -item["mentions"]))


def build_weekly_report(week_index, region):
    """Aggregate one region and calendar week into a reusable weekly brief."""
    calendar = _read_calendar()
    matches = calendar[calendar["week_index"] == int(week_index)]
    if matches.empty:
        raise ValueError(f"Reporting week {week_index} is not in the project calendar.")
    week_row = matches.iloc[0]

    stores = load_stores()
    if region not in set(stores["region"].dropna().astype(str)):
        raise ValueError(f"Unknown reporting region: {region}")

    feedback = _load_report_feedback()
    current = _period_feedback(feedback, week_row.week_start, week_row.week_end, region)
    if current.empty:
        raise ValueError(f"No feedback is available for {region} in week {week_index}.")

    earlier_weeks = calendar[calendar["week_start"] < week_row.week_start]
    prior_row = earlier_weeks.iloc[-1] if not earlier_weeks.empty else None
    previous = (
        _period_feedback(feedback, prior_row.week_start, prior_row.week_end, region)
        if prior_row is not None
        else feedback.iloc[0:0].copy()
    )

    current_actionable = current[
        pd.to_numeric(current["actionable"], errors="coerce").fillna(0).eq(1)
    ]
    previous_actionable = previous[
        pd.to_numeric(previous["actionable"], errors="coerce").fillna(0).eq(1)
    ]
    current_counts = _issue_counts(current_actionable)
    previous_counts = _issue_counts(previous_actionable)
    details = load_taxonomy()[["issue_type_id", "issue_type", "owner_function", "policy_severity_floor"]]
    current_counts = current_counts.merge(details, on="issue_type_id", how="left")
    current_counts["severity"] = current_counts["issue_type_id"].map(
        current.groupby("issue_type_id")["severity"].apply(
            lambda values: max(
                values.fillna("low").astype(str).str.lower(),
                key=lambda value: SEVERITY_ORDER.get(value, 0),
            )
        )
    )
    current_counts["severity_weight"] = current_counts["severity"].map(SEVERITY_WEIGHT).fillna(1.0)
    current_counts["rank_score"] = current_counts["mentions"] * current_counts["severity_weight"]
    current_counts = current_counts.merge(
        previous_counts[["issue_type_id", "mentions"]].rename(
            columns={"mentions": "previous_mentions"}
        ),
        on="issue_type_id",
        how="left",
    )
    current_counts["previous_mentions"] = current_counts["previous_mentions"].fillna(0).astype(int)
    current_counts["change"] = current_counts["mentions"] - current_counts["previous_mentions"]
    current_counts = current_counts.sort_values(
        ["rank_score", "mentions", "issue_type"], ascending=[False, False, True]
    )

    policy_watch = details[details["policy_severity_floor"].astype(str).str.lower().isin(
        POLICY_SEVERITIES
    )].merge(
        current_counts[
            ["issue_type_id", "mentions", "previous_mentions", "change", "stores"]
        ],
        on="issue_type_id",
        how="left",
    )
    policy_watch[["mentions", "previous_mentions", "change", "stores"]] = policy_watch[
        ["mentions", "previous_mentions", "change", "stores"]
    ].fillna(0).astype(int)
    policy_watch["severity_order"] = policy_watch["policy_severity_floor"].map(
        SEVERITY_ORDER
    ).fillna(0)
    policy_watch = policy_watch.sort_values(
        ["severity_order", "mentions", "issue_type"],
        ascending=[False, False, True],
    )

    baseline = None
    if prior_row is not None:
        current_mentions = len(current)
        prior_mentions = len(previous)
        change = current_mentions - prior_mentions
        change_percent = change / prior_mentions * 100 if prior_mentions else None
        baseline = {
            "week_index": int(prior_row.week_index),
            "week_start": prior_row.week_start.strftime("%Y-%m-%d"),
            "week_end": prior_row.week_end.strftime("%Y-%m-%d"),
            "mentions": prior_mentions,
            "stores": int(previous["store_id"].nunique()),
            "change": change,
            "change_percent": change_percent,
        }

    emerging_flags = _emerging_spikes(feedback, calendar, week_row, region, current)

    return {
        "week_index": int(week_row.week_index),
        "week_start": week_row.week_start.strftime("%Y-%m-%d"),
        "week_end": week_row.week_end.strftime("%Y-%m-%d"),
        "region": region,
        "generated_on": date.today().strftime("%d %b %Y"),
        "mentions": int(len(current)),
        "actionable": int(pd.to_numeric(current["actionable"], errors="coerce").fillna(0).sum()),
        "high_severity": int(current["severity"].fillna("low").astype(str).str.lower().isin(
            POLICY_SEVERITIES
        ).sum()),
        "stores": int(current["store_id"].nunique()),
        "baseline": baseline,
        "top_issues": current_counts.head(5).to_dict(orient="records"),
        "policy_watch": policy_watch.head(8).to_dict(orient="records"),
        "emerging_flags": emerging_flags[:5],
        "store_breakdown": _store_breakdown(current),
    }


def _store_breakdown(current):
    """Summarize weekly mentions, actionability, and severity for each active store."""
    if current.empty:
        return []
    current = current.copy()
    current["actionable"] = pd.to_numeric(current["actionable"], errors="coerce").fillna(0)
    current["severity"] = current["severity"].fillna("low").astype(str).str.lower()
    return (
        current.groupby(["store_id", "store_name"], dropna=False)
        .agg(
            mentions=("feedback_id", "count"),
            actionable=("actionable", "sum"),
            high_severity=("severity", lambda values: values.isin(POLICY_SEVERITIES).sum()),
        )
        .reset_index()
        .sort_values(["mentions", "store_name"], ascending=[False, True])
        .head(8)
        .to_dict(orient="records")
    )


def get_report_issue_evidence(week_index, region, store_id, issue_type_id):
    """Return original feedback supporting a store/category report spike."""
    calendar = _read_calendar()
    matches = calendar[calendar["week_index"] == int(week_index)]
    if matches.empty:
        raise ValueError(f"Reporting week {week_index} is not in the project calendar.")
    if region not in get_report_options()["regions"]:
        raise ValueError(f"Unknown reporting region: {region}")
    week = matches.iloc[0]
    feedback = _load_report_feedback()
    evidence = _period_feedback(feedback, week.week_start, week.week_end, region)
    evidence = evidence[
        (evidence["store_id"].astype(str) == str(store_id))
        & (evidence["issue_type_id"].astype(str) == str(issue_type_id))
    ].copy()
    if evidence.empty:
        raise ValueError("No matching feedback evidence was found for this report.")
    flag = {
        "issue_type": str(evidence.iloc[0]["issue_type"]),
        "issue_type_id": str(issue_type_id),
        "store_id": str(store_id),
        "store_name": str(evidence.iloc[0].get("store_name") or store_id),
        "owner_function": str(evidence.iloc[0].get("owner_function") or "Unassigned"),
        "week_index": int(week_index),
        "week_start": week.week_start.strftime("%Y-%m-%d"),
        "week_end": week.week_end.strftime("%Y-%m-%d"),
    }
    return {
        "flag": flag,
        "evidence": evidence.sort_values("created_at").to_dict(orient="records"),
    }


def _report_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportKicker", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7, leading=9, textColor=PURPLE, tracking=1.1,
    ))
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=26, alignment=TA_LEFT, textColor=INK, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="SectionLabel", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.1, leading=9, textColor=INK, tracking=.35, spaceBefore=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["Normal"], fontSize=7, leading=9,
        textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="ReportSmall", parent=styles["Normal"], fontSize=6.5, leading=8,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="KpiLabel", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=6.2, leading=8, textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="KpiValue", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=17, leading=19, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="Cell", parent=styles["Normal"], fontSize=6.4, leading=8,
        textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="CellBold", parent=styles["Cell"], fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="CellMuted", parent=styles["Cell"], textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="CellHeader", parent=styles["Cell"], fontName="Helvetica-Bold",
        fontSize=5.8, leading=7, textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="CellRight", parent=styles["Cell"], alignment=TA_RIGHT,
    ))
    styles.add(ParagraphStyle(
        name="FooterText", parent=styles["Normal"], fontSize=6, leading=7,
        textColor=MUTED,
    ))
    return styles


def _paragraph(value, style, max_length=None):
    text = "" if value is None or pd.isna(value) else str(value)
    if max_length and len(text) > max_length:
        text = text[:max_length - 1] + "…"
    return Paragraph(escape(text), style)


def _signed_change(change, prior):
    change = int(change)
    if change == 0:
        return "flat"
    percent = f"{abs(change / prior * 100):.0f}%" if prior else "new"
    sign = "+" if change > 0 else "-"
    return f"{sign}{abs(change)} ({sign}{percent})"


def _draw_footer(canvas, document):
    canvas.saveState()
    page_width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(15 * mm, 13 * mm, page_width - 15 * mm, 13 * mm)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        15 * mm,
        9 * mm,
        f"StorePulse Intelligence · {document.region} region · week {document.week_index}",
    )
    canvas.drawRightString(page_width - 15 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def generate_weekly_pdf(report):
    """Render a one-page, downloadable PDF in the style of the supplied brief."""
    output = BytesIO()
    page_width, page_height = A4
    margin_x = 12 * mm
    margin_top = 12 * mm
    margin_bottom = 16 * mm
    document = BaseDocTemplate(
        output,
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
        title=f"StorePulse {report['region']} week {report['week_index']} brief",
        author="StorePulse Intelligence",
    )
    document.region = report["region"]
    document.week_index = report["week_index"]
    frame = Frame(
        margin_x,
        margin_bottom,
        page_width - 2 * margin_x,
        page_height - margin_top - margin_bottom,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates(
        [PageTemplate(id="brief", frames=[frame], onPage=_draw_footer)]
    )
    styles = _report_styles()
    story = []
    date_range = (
        f"{pd.Timestamp(report['week_start']).strftime('%d %b')} – "
        f"{pd.Timestamp(report['week_end']).strftime('%d %b %Y')}"
    )
    story.extend([
        _paragraph("STOREPULSE · WEEKLY REGIONAL BRIEF", styles["ReportKicker"]),
        Spacer(1, 2),
        _paragraph(f"{escape(report['region'])} region", styles["ReportTitle"]),
        _paragraph(
            f"Week {report['week_index']} · {date_range} · "
            f"{report['stores']} stores reporting · generated {report['generated_on']}",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 8),
    ])

    kpi_cards = []
    for label, value in (
        ("MENTIONS", report["mentions"]),
        ("ACTIONABLE", report["actionable"]),
        ("HIGH SEVERITY", report["high_severity"]),
        ("STORES AFFECTED", report["stores"]),
    ):
        kpi_cards.append([
            _paragraph(label, styles["KpiLabel"]),
            _paragraph(f"{value:,}", styles["KpiValue"]),
        ])
    kpis = Table(
        [[
            Table(
                [[card[0]], [card[1]]],
                colWidths=[(page_width - 2 * margin_x - 21 * mm) / 4 - 9 * mm],
            )
            for card in kpi_cards
        ]],
        colWidths=[(page_width - 2 * margin_x) / 4] * 4,
    )
    kpis.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), .6, LINE),
        ("INNERGRID", (0, 0), (-1, -1), .6, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([kpis, Spacer(1, 5)])

    story.append(_paragraph("BASELINE CHECK", styles["SectionLabel"]))
    baseline = report["baseline"]
    if baseline:
        change = _signed_change(baseline["change"], baseline["mentions"])
        baseline_text = (
            f"{report['mentions']:,} mentions this week, {change} against the prior-week "
            f"baseline. Prior week closed on {baseline['mentions']:,} mentions across "
            f"{baseline['stores']} stores."
        )
    else:
        baseline_text = "No earlier calendar week is available as a comparison baseline."
    policy_note = (
        "Policy floors stay visible regardless of mention volume; store spikes compare "
        "this week with each store/category’s prior eight-week average."
    )
    story.extend([
        _paragraph(baseline_text, styles["ReportBody"]),
        _paragraph(policy_note, styles["ReportSmall"]),
        Spacer(1, 4),
    ])

    story.append(_paragraph("TOP ISSUES THIS WEEK", styles["SectionLabel"]))
    top_data = [[
        _paragraph("#", styles["CellHeader"]),
        _paragraph("ISSUE", styles["CellHeader"]),
        _paragraph("OWNER", styles["CellHeader"]),
        _paragraph("SEVERITY", styles["CellHeader"]),
        _paragraph("THIS WK", styles["CellHeader"]),
        _paragraph("PREV", styles["CellHeader"]),
        _paragraph("CHANGE", styles["CellHeader"]),
        _paragraph("STORES", styles["CellHeader"]),
    ]]
    for rank, issue in enumerate(report["top_issues"], start=1):
        severity = str(issue["severity"]).lower()
        severity_style = ParagraphStyle(
            f"severity_{rank}", parent=styles["CellBold"],
            textColor=SEVERITY_COLORS.get(severity, INK),
        )
        top_data.append([
            _paragraph(rank, styles["CellMuted"]),
            _paragraph(issue["issue_type"], styles["CellBold"], 66),
            _paragraph(issue["owner_function"], styles["CellMuted"], 22),
            _paragraph(severity.title(), severity_style),
            _paragraph(f"{int(issue['mentions']):,}", styles["CellRight"]),
            _paragraph(f"{int(issue['previous_mentions']):,}", styles["CellRight"]),
            _paragraph(
                _signed_change(issue["change"], issue["previous_mentions"]),
                styles["CellRight"],
            ),
            _paragraph(f"{int(issue['stores']):,}", styles["CellRight"]),
        ])
    top_table = Table(
        top_data,
        colWidths=[6 * mm, 64 * mm, 28 * mm, 18 * mm, 14 * mm, 13 * mm, 22 * mm, 12 * mm],
        repeatRows=1,
    )
    top_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEBELOW", (0, 0), (-1, 0), .6, LINE),
        ("LINEBELOW", (0, 1), (-1, -1), .35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.extend([top_table, Spacer(1, 3)])

    story.append(_paragraph("STORE BREAKDOWN", styles["SectionLabel"]))
    store_data = [[
        _paragraph("STORE", styles["CellHeader"]),
        _paragraph("MENTIONS", styles["CellHeader"]),
        _paragraph("ACTIONABLE", styles["CellHeader"]),
        _paragraph("HIGH SEV.", styles["CellHeader"]),
    ]]
    for store in report["store_breakdown"][:5]:
        store_data.append([
            _paragraph(store.get("store_name") or store.get("store_id"), styles["Cell"], 50),
            _paragraph(int(store["mentions"]), styles["CellRight"]),
            _paragraph(int(store["actionable"]), styles["CellRight"]),
            _paragraph(int(store["high_severity"]), styles["CellRight"]),
        ])
    stores_table = Table(store_data, colWidths=[92 * mm, 28 * mm, 30 * mm, 26 * mm], repeatRows=1)
    stores_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("LINEBELOW", (0, 0), (-1, 0), .5, LINE),
        ("LINEBELOW", (0, 1), (-1, -1), .25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.extend([stores_table, Spacer(1, 3)])

    story.append(_paragraph("POLICY WATCH · NEVER DEPRIORITISED", styles["SectionLabel"]))
    policy_data = []
    for issue in report["policy_watch"]:
        prior_mentions = int(issue["previous_mentions"])
        current_mentions = int(issue["mentions"])
        change = int(issue["change"])
        policy_data.append([
            _paragraph(issue["issue_type"], styles["CellBold"], 72),
            _paragraph(f"{current_mentions} this wk", styles["CellRight"]),
            _paragraph(f"{prior_mentions} prev", styles["CellRight"]),
            _paragraph(
                f"{'+' if change > 0 else ''}{change}" if change else "flat",
                styles["CellRight"],
            ),
            _paragraph(f"floor {str(issue['policy_severity_floor']).title()}", styles["CellRight"]),
        ])
    if not policy_data:
        policy_data = [[_paragraph("No high/critical policy floors are configured.", styles["CellMuted"])]]
        policy_widths = [page_width - 2 * margin_x]
    else:
        policy_widths = [78 * mm, 25 * mm, 22 * mm, 18 * mm, 34 * mm]
    policy_table = Table(policy_data, colWidths=policy_widths)
    policy_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), .3, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.extend([policy_table, Spacer(1, 3)])

    story.append(_paragraph("EMERGING FLAGS · EARLY WARNING", styles["SectionLabel"]))
    if report["emerging_flags"]:
        flag_data = []
        for flag in report["emerging_flags"]:
            flag_data.append([
                _paragraph(
                    f"{flag['store_name']} · {flag['issue_type']}",
                    styles["CellBold"],
                    85,
                ),
                _paragraph(
                    f"z {flag['z_score']:.1f} · {flag['lift']:.1f}× lift · "
                    f"{flag['mentions']} this week vs {flag['baseline']:.1f} 8-week mean · "
                    f"floor {flag['policy_severity_floor']}",
                    styles["CellMuted"],
                ),
            ])
        flags_table = Table(flag_data, colWidths=[78 * mm, page_width - 2 * margin_x - 78 * mm])
        flags_table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), .3, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(flags_table)
    else:
        story.append(_paragraph(
            "No store/category spikes met the minimum evidence, 3σ and 2× lift thresholds this week.",
            styles["CellMuted"],
        ))

    story.extend([
        Spacer(1, 4),
        _paragraph(
            "HOW THIS BRIEF IS BUILT",
            styles["SectionLabel"],
        ),
        _paragraph(
            "Rank = mentions × severity weight (critical 2.4, high 1.8, medium 1.35, low 1.0). "
            "Model predictions are raised to taxonomy policy floors. Store spikes compare "
            "weekly store/category counts with the prior eight calendar weeks; a flag requires "
            "at least 3 mentions, z ≥ 3 and lift ≥ 2×.",
            styles["FooterText"],
        ),
    ])
    document.build(story)
    output.seek(0)
    return output
