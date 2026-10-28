import os
import pandas as pd

os.environ["PYTHONPATH"] = os.path.abspath("FULL_CODE/pipeline/src")

from voice_of_store.predict import predict_records

data_dir = "FULL_CODE/db"
feedback = pd.read_csv(f"{data_dir}/feedback.csv").fillna("")
labels = pd.read_csv(f"{data_dir}/feedback_labels_train.csv").fillna("")
taxonomy = pd.read_csv(f"{data_dir}/issue_taxonomy.csv").fillna("")

labelled = labels.merge(
    feedback[["feedback_id", "text", "store_id", "created_at"]],
    on="feedback_id",
    how="left",
    validate="one_to_one"
)

print("Labelled rows:", len(labelled))
print("Rows with missing text:", labelled["text"].eq("").sum())

print("\nIssue categories in training labels:")
print(labelled["issue_type_id"].value_counts().to_string())

print("\nSafety-related labelled examples:")
safety_ids = {"IT04", "IT06", "IT08", "IT10", "IT16", "IT17", "IT18"}
safety = labelled[labelled["issue_type_id"].isin(safety_ids)]
print(safety[["feedback_id", "issue_type_id", "severity", "text"]].head(12).to_string(index=False))
