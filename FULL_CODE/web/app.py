import json
import os
import secrets
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from config import (
    APP_HOST,
    APP_PORT,
    DEBUG,
    FEEDBACK_PAGE_SIZE,
    MAX_UPLOAD_BYTES,
    MAX_FEEDBACK_PAGE_SIZE,
)
from services.data_service import get_overview, get_emerging_issues, get_issue_details
from services.feedback_service import get_feedback, get_filter_options
from services.ingestion_service import import_feedback_upload, preview_feedback_upload
from services.issue_service import get_managed_issues, set_managed_issue_status
from services.report_service import (
    build_weekly_report,
    get_report_issue_evidence,
    generate_weekly_pdf,
    get_report_options,
)
from services.translation_service import (
    TranslationUnavailableError,
    get_translation_languages,
    translate_text,
)
from services.data_service import load_stores
from services.storage_service import get_import_history, initialize_storage

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "STOREPULSE_SECRET_KEY", secrets.token_hex(32)
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

initialize_storage()


def _available_regions():
    stores = load_stores()
    if "region" not in stores.columns:
        raise ValueError("store_master.csv is missing the required region column.")
    return sorted(stores["region"].dropna().astype(str).unique().tolist())


def _active_region():
    regions = _available_regions()
    if not regions:
        raise ValueError("No regions are available in store_master.csv.")
    region = session.get("region", "Central")
    if region not in regions:
        region = "Central" if "Central" in regions else regions[0]
        session["region"] = region
    return region


@app.context_processor
def inject_workspace_context():
    return {
        "workspace_regions": _available_regions(),
        "current_region": _active_region(),
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."}), 413
    return render_template(
        "error.html",
        error=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
    ), 413


def _json_records(frame):
    """Serialize dataframe records with standards-compliant nulls and timestamps."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _template_records(frame):
    """Keep pandas timestamps usable by Jinja while normalizing missing values."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


def _feedback_filters():
    return {
        "search": request.args.get("search", "").strip(),
        "channel": request.args.get("channel", "").strip(),
        "store_id": request.args.get("store_id", "").strip(),
        "severity": request.args.get("severity", "").strip(),
        "actionable": request.args.get("actionable", "").strip(),
        "issue_type_id": request.args.get("issue_type_id", "").strip(),
        "region": request.args.get("region", _active_region()).strip(),
    }


@app.route("/")
def index():
    return redirect(url_for("reports_page"))


@app.route("/set-region")
def set_region():
    region = request.args.get("region", "").strip()
    if region not in _available_regions():
        abort(400, description="Choose a region from the store master list.")
    session["region"] = region
    next_path = request.args.get("next", "/reports")
    parsed = urlsplit(next_path)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/") or parsed.path.startswith("//"):
        next_path = "/reports"
    if parsed.query:
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["region"] = region
        next_path = urlunsplit(("", "", parsed.path, urlencode(query), ""))
    else:
        next_path = parsed.path
    return redirect(next_path)


@app.route("/feedback")
def feedback_page():
    selected = _feedback_filters()
    records = get_feedback(**selected)

    filters = get_filter_options()
    page = max(1, request.args.get("page", 1, type=int))
    per_page = request.args.get("per_page", FEEDBACK_PAGE_SIZE, type=int)
    if not 1 <= per_page <= MAX_FEEDBACK_PAGE_SIZE:
        per_page = FEEDBACK_PAGE_SIZE
    total = len(records)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page

    return render_template(
        "feedback.html",
        records=_template_records(records.iloc[start:start + per_page]),
        total=total,
        filters=filters,
        selected=selected,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@app.route("/emerging")
def emerging_page():
    options = get_report_options()
    selected_week = request.args.get("week", options["weeks"][-1]["week_index"], type=int)
    selected_region = request.args.get("region", _active_region()).strip()
    selected_status = request.args.get("status", "all").strip().lower()
    if selected_status not in {"all", "new", "investigating", "archived"}:
        selected_status = "all"
    try:
        flags = get_managed_issues(selected_week, selected_region)
        error = None
    except ValueError as exc:
        flags = []
        error = str(exc)
    records = [
        flag for flag in flags
        if selected_status == "all" or flag["status"] == selected_status
    ]
    return render_template(
        "emerging.html",
        records=records,
        weeks=options["weeks"],
        selected={
            "week": selected_week,
            "region": selected_region,
            "status": selected_status,
        },
        error=error,
    )


@app.route("/issue/<int:flag_index>")
def issue_detail(flag_index):
    details = get_issue_details(flag_index)
    if details is None:
        return "Emerging issue not found", 404

    return render_template(
        "issue_detail.html",
        flag=details["flag"],
        evidence=details["evidence"],
    )


@app.route("/reports")
def reports_page():
    options = get_report_options()
    if not options["weeks"]:
        return render_template(
            "report.html",
            options=options,
            report=None,
            error="No weeks are available in the reporting calendar.",
        ), 500

    default_week = options["weeks"][-1]["week_index"]
    week_index = request.args.get("week", default_week, type=int)
    if week_index is None:
        week_index = default_week
    default_region = _active_region()
    region = request.args.get("region", default_region).strip()
    if region in options["regions"]:
        session["region"] = region
    try:
        report = build_weekly_report(week_index, region)
        error = None
    except ValueError as exc:
        report = None
        error = str(exc)

    return render_template(
        "report.html",
        options=options,
        report=report,
        error=error,
        selected={"week": week_index, "region": region},
    )


@app.route("/data")
def data_page():
    return render_template("data.html", imports=get_import_history())


@app.route("/reports/<int:week_index>/<region>/pdf")
def weekly_report_pdf(week_index, region):
    try:
        report = build_weekly_report(week_index, region)
        pdf = generate_weekly_pdf(report)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    filename = (
        f"storepulse-{region.lower().replace(' ', '-')}"
        f"-week-{week_index}-{report['week_start'][:4]}.pdf"
    )
    return send_file(
        pdf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/reports/<int:week_index>/<region>/store/<store_id>/issue/<issue_type_id>")
def report_issue_detail(week_index, region, store_id, issue_type_id):
    try:
        details = get_report_issue_evidence(
            week_index, region, store_id, issue_type_id
        )
    except ValueError as exc:
        return render_template("error.html", error=str(exc)), 404
    return render_template(
        "issue_detail.html",
        flag=details["flag"],
        evidence=details["evidence"],
    )


@app.route("/api/overview")
def overview_api():
    return jsonify(get_overview())


@app.route("/api/feedback")
def feedback_api():
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", FEEDBACK_PAGE_SIZE))
    except ValueError:
        return jsonify({"error": "page and per_page must be integers."}), 400
    if page < 1 or not 1 <= per_page <= MAX_FEEDBACK_PAGE_SIZE:
        return jsonify({
            "error": f"page must be positive and per_page must be between 1 and "
            f"{MAX_FEEDBACK_PAGE_SIZE}."
        }), 400

    records = get_feedback(**_feedback_filters())
    total = len(records)
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > total_pages:
        return jsonify({"error": "page is past the end of the filtered results."}), 400
    start = (page - 1) * per_page
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "records": _json_records(records.iloc[start:start + per_page]),
    })


@app.route("/api/emerging")
def emerging_api():
    flags = get_emerging_issues()
    return jsonify({"records": _json_records(flags)})


@app.route("/api/issues/<int:flag_index>")
def issue_api(flag_index):
    details = get_issue_details(flag_index)
    if details is None:
        return jsonify({"error": "Emerging issue not found."}), 404

    flag = details["flag"]
    return jsonify({
        "flag": _json_records(pd.DataFrame([flag]))[0],
        "evidence": _json_records(pd.DataFrame(details["evidence"])),
    })


@app.route("/api/emerging/status", methods=["POST"])
def emerging_status_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required."}), 400
    issue_key = payload.get("issue_key")
    status = payload.get("status")
    week_index = payload.get("week")
    region = payload.get("region", _active_region())
    if not isinstance(issue_key, str) or not issue_key or len(issue_key) > 160:
        return jsonify({"error": "A valid issue_key is required."}), 400
    if not isinstance(status, str) or not isinstance(week_index, int):
        return jsonify({"error": "status and integer week are required."}), 400
    try:
        updated_status = set_managed_issue_status(
            issue_key, status, week_index, str(region)
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"issue_key": issue_key, "status": updated_status})


@app.route("/api/translation/languages")
def translation_languages_api():
    try:
        return jsonify(get_translation_languages())
    except TranslationUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/data/preview", methods=["POST"])
def data_preview_api():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a CSV file to preview."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV feedback files are supported."}), 400
    try:
        preview = preview_feedback_upload(upload.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(preview)


@app.route("/api/data/upload", methods=["POST"])
def data_upload_api():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a CSV file to import."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV feedback files are supported."}), 400
    try:
        result = import_feedback_upload(
            upload.read(), secure_filename(upload.filename)
        )
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    if result["records_added"] == 0:
        return jsonify({
            "error": "Every feedback ID in this upload already exists.",
            **result,
        }), 409
    return jsonify({
        **result,
        "message": "New feedback was classified and appended to local SQLite storage.",
    }), 201


@app.route("/api/translate", methods=["POST"])
def translate_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "A JSON request body is required."}), 400

    text = payload.get("text")
    source_language = payload.get("source_language")
    target_language = payload.get("target_language")
    if not all(isinstance(value, str) for value in (text, source_language, target_language)):
        return jsonify({
            "error": "text, source_language, and target_language must be strings."
        }), 400
    if len(text) > 5000:
        return jsonify({"error": "Text must be 5,000 characters or fewer."}), 400

    try:
        translated_text = translate_text(text, source_language, target_language)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except TranslationUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({
        "translation": translated_text,
        "source_language": source_language,
        "target_language": target_language,
        "engine": "Argos Translate (local)",
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "StorePulse Intelligence"})


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=DEBUG)
