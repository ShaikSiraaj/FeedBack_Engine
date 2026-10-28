import sys
from pathlib import Path
import pandas as pd

root = Path.cwd()
sys.path.insert(0, str(root / "FULL_CODE" / "pipeline" / "src"))

from voice_of_store.features import load_training_data
from voice_of_store.train import _new_classifier, stratified_indices

data, taxonomy = load_training_data(root / "FULL_CODE" / "db")
data = data.reset_index(drop=True)
train_idx, val_idx = stratified_indices(data["issue_type_id"], seed=42)

train = data.iloc[train_idx].copy()
val = data.iloc[val_idx].copy()

model = _new_classifier(seed=42)
model.fit(train["text"].fillna("").astype(str), train["issue_type_id"].astype(str))
val["predicted_issue_type_id"] = model.predict(val["text"].fillna("").astype(str))

safety_ids = {"IT04", "IT06", "IT08", "IT10", "IT16", "IT17", "IT18"}
terms = [
    "food poisoning", "vomiting", "fell sick", "allergic", "allergy",
    "rash", "metal pin", "foreign object", "glass in", "expired",
    "expiry", "rotten", "cockroach", "rat droppings", "flies",
    "warm freezer", "fridge was", "chiller", "wet floor",
    "caution board", "no caution", "contamination"
]

val["text_lower"] = val["text"].fillna("").astype(str).str.lower()
val["text_safety_signal"] = val["text_lower"].apply(
    lambda t: any(term in t for term in terms)
)
val["predicted_safety_category"] = val["predicted_issue_type_id"].isin(safety_ids)

print("Validation rows:", len(val))
print("Rows with one of the explicit safety phrases:", int(val["text_safety_signal"].sum()))
print("Safety-phrase rows predicted into a high-floor category:",
      int((val["text_safety_signal"] & val["predicted_safety_category"]).sum()))
print("Safety-phrase rows predicted outside a high-floor category:",
      int((val["text_safety_signal"] & ~val["predicted_safety_category"]).sum()))

print("\n=== Safety-phrase rows predicted outside high-floor categories ===")
cols = ["feedback_id", "issue_type_id", "predicted_issue_type_id", "severity", "text"]
print(
    val.loc[val["text_safety_signal"] & ~val["predicted_safety_category"], cols]
       .head(40).to_string(index=False)
)

out = root / "FULL_CODE" / "submission" / "R5_validation_safety_signals.csv"
val.loc[val["text_safety_signal"], cols].to_csv(out, index=False)
print("\nSaved:", out)
