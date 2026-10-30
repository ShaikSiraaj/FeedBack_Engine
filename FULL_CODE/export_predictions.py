from pathlib import Path
import pandas as pd
import detect

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "db"
MODEL_PATH = ROOT / "models" / "r5_classifier.joblib"
OUTPUT_PATH = ROOT / "submission" / "R5_predictions.csv"

feedback = pd.read_csv(DATA_DIR / "feedback.csv", low_memory=False)

predictions = detect.predict_feedback(feedback, MODEL_PATH)

required_columns = [
    "feedback_id",
    "issue_type_id",
    "severity",
    "actionable",
]
predictions = predictions[required_columns]

if predictions["feedback_id"].duplicated().any():
    raise ValueError("Duplicate feedback_id values found in predictions.")

if len(predictions) != len(feedback):
    raise ValueError(
        f"Expected {len(feedback)} predictions, got {len(predictions)}."
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT_PATH, index=False)

print(f"Feedback rows: {len(feedback)}")
print(f"Predictions exported: {len(predictions)}")
print(f"Saved to: {OUTPUT_PATH}")
print(predictions.head().to_string(index=False))
