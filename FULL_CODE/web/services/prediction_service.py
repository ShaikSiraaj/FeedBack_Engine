from functools import lru_cache

import joblib
import pandas as pd

from config import FEEDBACK_FILE, MODEL_FILE
from services.data_service import load_feedback

import sys
from pathlib import Path

PIPELINE_SRC = Path(__file__).resolve().parents[2] / "pipeline" / "src"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from voice_of_store.predict import predict_records


def predict_feedback_records(feedback):
    """Run the currently configured classifier on an additional feedback batch."""
    return predict_records(feedback, load_model_bundle())


@lru_cache(maxsize=1)
def load_model_bundle():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


@lru_cache(maxsize=1)
def get_predictions():
    feedback = _load_original_feedback()
    bundle = load_model_bundle()

    batch_size = 2000
    batches = []

    for start in range(0, len(feedback), batch_size):
        batch = feedback.iloc[start:start + batch_size].copy()
        batches.append(predict_records(batch, bundle))

    if not batches:
        import pandas as pd
        return pd.DataFrame(
            columns=["feedback_id", "issue_type_id", "severity", "actionable"]
        )

    import pandas as pd
    return pd.concat(batches, ignore_index=True)


def get_feedback_with_predictions():
    feedback = load_feedback()
    predictions = get_predictions()

    original_feedback = _load_original_feedback()
    original_records = original_feedback.merge(
        predictions,
        on="feedback_id",
        how="left",
        validate="one_to_one",
    )
    from services.storage_service import load_uploaded_feedback

    uploaded = load_uploaded_feedback()
    if uploaded.empty:
        return original_records

    return pd.concat([original_records, uploaded], ignore_index=True).sort_values(
        "created_at", ascending=False, na_position="last"
    )


def _load_original_feedback():
    """Load unaugmented source CSV rows so database imports can be de-duplicated."""
    feedback = pd.read_csv(
        FEEDBACK_FILE,
        encoding="utf-8-sig",
        low_memory=False,
        parse_dates=["created_at"],
    )
    feedback["store_id"] = feedback["store_id"].fillna("").astype(str)
    feedback["text"] = feedback["text"].fillna("").astype(str)
    return feedback
