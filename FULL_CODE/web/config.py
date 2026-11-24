from pathlib import Path

# Project layout:
# TEAM-001392/
#   FULL_CODE/
#     db/
#     models/
#     submission/
#     web/
#       config.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_CODE_DIR = PROJECT_ROOT / "FULL_CODE"

DB_DIR = FULL_CODE_DIR / "db"
MODEL_DIR = FULL_CODE_DIR / "models"
SUBMISSION_DIR = FULL_CODE_DIR / "submission"
STORAGE_FILE = FULL_CODE_DIR / "storepulse.sqlite3"

FEEDBACK_FILE = DB_DIR / "feedback.csv"
TAXONOMY_FILE = DB_DIR / "issue_taxonomy.csv"
STORE_FILE = DB_DIR / "store_master.csv"
CALENDAR_FILE = DB_DIR / "week_calendar.csv"

MODEL_FILE = MODEL_DIR / "r5_classifier.joblib"

EMERGING_FLAGS_FILE = SUBMISSION_DIR / "R5_emerging_flags.csv"
FEEDBACK_PAGE_SIZE = 25
MAX_FEEDBACK_PAGE_SIZE = 100
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_RECORDS = 10000

# Dashboard settings
APP_HOST = "127.0.0.1"
APP_PORT = 5000
DEBUG = True

# Do not overwrite or modify the original feedback text.
DEFAULT_LANGUAGE_MODE = "original"