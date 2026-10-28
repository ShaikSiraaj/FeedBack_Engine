import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "FULL_CODE" / "pipeline" / "src"))

from voice_of_store.features import load_training_data
from voice_of_store.train import _new_classifier, stratified_indices

DATA_DIR = ROOT / "FULL_CODE" / "db"
OUT_DIR = ROOT / "FULL_CODE" / "submission"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data, taxonomy = load_training_data(DATA_DIR)
data = data.reset_index(drop=True)

train_idx, val_idx = stratified_indices(data["issue_type_id"], seed=42)
train = data.iloc[train_idx].copy()
val = data.iloc[val_idx].copy()

print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(val)}")

targets = ["issue_type_id", "severity", "actionable"]
reports = []

for target in targets:
    model = _new_classifier(seed=42)
    model.fit(train["text"].fillna("").astype(str), train[target].astype(str))
    pred = model.predict(val["text"].fillna("").astype(str))

    actual = val[target].astype(str).reset_index(drop=True)
    pred = pd.Series(pred, dtype=str)

    report = classification_report(
        actual, pred, output_dict=True, zero_division=0
    )
    print(f"\n{'=' * 20} {target} {'=' * 20}")
    print(classification_report(actual, pred, zero_division=0))

    for label, metrics in report.items():
        if isinstance(metrics, dict) and label not in ("macro avg", "weighted avg"):
            reports.append({
                "target": target,
                "label": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1-score"],
                "support": metrics["support"],
            })

    if target == "issue_type_id":
        errors = val[["feedback_id", "text", target]].copy()
        errors = errors.rename(columns={target: "actual"})
        errors["predicted"] = pred.values
        errors = errors[errors["actual"] != errors["predicted"]]
        errors.to_csv(OUT_DIR / "R5_validation_issue_errors.csv", index=False)

        print("\nIssue category confusion matrix (largest confusions):")
        cm = confusion_matrix(actual, pred, labels=model.classes_)
        cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
        confusion_pairs = []
        for i, actual_label in enumerate(model.classes_):
            for j, predicted_label in enumerate(model.classes_):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append(
                        (int(cm[i, j]), actual_label, predicted_label)
                    )
        for count, a, p in sorted(confusion_pairs, reverse=True)[:20]:
            print(f"Actual {a} -> predicted {p}: {count}")

    elif target == "severity":
        errors = val[["feedback_id", "text", target]].copy()
        errors = errors.rename(columns={target: "actual"})
        errors["predicted"] = pred.values
        errors = errors[errors["actual"] != errors["predicted"]]
        errors.to_csv(OUT_DIR / "R5_validation_severity_errors.csv", index=False)

pd.DataFrame(reports).to_csv(
    OUT_DIR / "R5_validation_classification_report.csv", index=False
)

print("\nSaved reports:")
print(OUT_DIR / "R5_validation_classification_report.csv")
print(OUT_DIR / "R5_validation_issue_errors.csv")
print(OUT_DIR / "R5_validation_severity_errors.csv")
