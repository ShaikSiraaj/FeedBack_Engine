import pandas as pd
from pathlib import Path

root = Path.cwd()
db = root / "FULL_CODE" / "db"
sub = root / "FULL_CODE" / "submission"

feedback = pd.read_csv(db / "feedback.csv").fillna("")
labels = pd.read_csv(db / "feedback_labels_train.csv").fillna("")

severity_file = sub / "R5_validation_severity_errors.csv"
errors = pd.read_csv(severity_file).fillna("")

# Reconstruct the exact validation split so we can identify critical cases
import sys
sys.path.insert(0, str(root / "FULL_CODE" / "pipeline" / "src"))
from voice_of_store.features import load_training_data
from voice_of_store.train import stratified_indices, _new_classifier

data, taxonomy = load_training_data(db)
data = data.reset_index(drop=True)
train_idx, val_idx = stratified_indices(data["issue_type_id"], seed=42)
train = data.iloc[train_idx].copy()
val = data.iloc[val_idx].copy()

model = _new_classifier(seed=42)
model.fit(train["text"].fillna("").astype(str), train["severity"].astype(str))
val["predicted_severity"] = model.predict(val["text"].fillna("").astype(str))

print("\n=== All critical validation examples ===")
critical = val[val["severity"].astype(str).str.lower() == "critical"]
print(critical[["feedback_id", "severity", "predicted_severity", "issue_type_id", "text"]]
      .to_string(index=False))

print("\n=== High severity predicted as low or medium ===")
missed_high = val[
    (val["severity"].astype(str).str.lower() == "high") &
    (val["predicted_severity"].astype(str).str.lower().isin(["low", "medium"]))
]
print(missed_high[["feedback_id", "severity", "predicted_severity", "issue_type_id", "text"]]
      .head(30).to_string(index=False))

print("\n=== Taxonomy policy severity floors ===")
cols = ["issue_type_id", "issue_type", "policy_severity_floor", "policy_reason"]
print(taxonomy[cols].to_string(index=False))
