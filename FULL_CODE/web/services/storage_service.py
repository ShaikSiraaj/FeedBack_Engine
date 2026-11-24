"""Local SQLite storage for uploaded feedback and emerging issue statuses."""

import sqlite3

import pandas as pd

from config import STORAGE_FILE


def _connect():
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORAGE_FILE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_storage():
    """Create the local upload and issue-status tables if needed."""
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploaded_feedback (
                feedback_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                created_at TEXT NOT NULL,
                store_id TEXT NOT NULL,
                rating REAL,
                text TEXT NOT NULL,
                issue_type_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                actionable INTEGER NOT NULL CHECK (actionable IN (0, 1)),
                source_filename TEXT NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_uploaded_feedback_created_at
                ON uploaded_feedback (created_at);
            CREATE INDEX IF NOT EXISTS idx_uploaded_feedback_store
                ON uploaded_feedback (store_id);
            CREATE TABLE IF NOT EXISTS feedback_imports (
                import_id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                records_added INTEGER NOT NULL,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS emerging_issue_statuses (
                issue_key TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (
                    status IN ('new', 'investigating', 'archived')
                ),
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def load_uploaded_feedback():
    """Read appended feedback records, leaving original CSV files unchanged."""
    initialize_storage()
    with _connect() as connection:
        frame = pd.read_sql_query(
            """
            SELECT feedback_id, channel, created_at, store_id, rating, text,
                   issue_type_id, severity, actionable
            FROM uploaded_feedback
            ORDER BY created_at, feedback_id
            """,
            connection,
        )
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
        frame["actionable"] = pd.to_numeric(frame["actionable"], errors="coerce").astype(int)
    return frame


def save_uploaded_feedback(records, filename):
    """Append validated, model-enriched records in one SQLite transaction."""
    initialize_storage()
    rows = [
        (
            str(record["feedback_id"]),
            str(record["channel"]),
            pd.Timestamp(record["created_at"]).isoformat(),
            str(record["store_id"]),
            None if pd.isna(record["rating"]) else float(record["rating"]),
            str(record["text"]),
            str(record["issue_type_id"]),
            str(record["severity"]),
            int(record["actionable"]),
            filename,
        )
        for record in records
    ]
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO uploaded_feedback (
                feedback_id, channel, created_at, store_id, rating, text,
                issue_type_id, severity, actionable, source_filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute(
            "INSERT INTO feedback_imports (filename, records_added) VALUES (?, ?)",
            (filename, len(rows)),
        )
    return len(rows)


def get_import_history(limit=20):
    """Return recent upload receipts without exposing feedback text."""
    initialize_storage()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT import_id, filename, records_added, imported_at
            FROM feedback_imports ORDER BY imported_at DESC, import_id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def load_issue_statuses():
    """Read persisted triage state for emerging issues."""
    initialize_storage()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT issue_key, status, updated_at FROM emerging_issue_statuses"
        ).fetchall()
    return {row["issue_key"]: dict(row) for row in rows}


def update_issue_status(issue_key, status):
    """Persist an allowed status transition for one detector issue."""
    if status not in {"new", "investigating", "archived"}:
        raise ValueError("Status must be new, investigating, or archived.")
    initialize_storage()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO emerging_issue_statuses (issue_key, status, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(issue_key) DO UPDATE SET
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
            """,
            (issue_key, status),
        )
