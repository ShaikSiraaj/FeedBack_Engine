from pathlib import Path
import pandas as pd

root = Path("FULL_CODE")
pred_path = root / "submission" / "R5_predictions.csv"

if not pred_path.exists():
    raise FileNotFoundError(f"Missing predictions file: {pred_path}")

pred = pd.read_csv(pred_path)
feedback = pd.read_csv(root / "db" / "feedback.csv")

df = feedback.merge(pred, on="feedback_id", how="left", validate="one_to_one")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["week_start"] = df["created_at"].dt.to_period("W-SUN").dt.start_time

week = pd.Timestamp("2026-05-11")
df = df[df["week_start"] == week].copy()

grouped = (
    df.groupby(["store_id", "issue_type_id"], dropna=True)
      .agg(
          current_count=("feedback_id", "nunique"),
          high_count=("severity", lambda s: s.astype(str).str.lower().isin(["high", "critical"]).sum())
      )
      .reset_index()
)

grouped = grouped[grouped["current_count"] >= 2].copy()
grouped["volume_score"] = (grouped["current_count"] / 5.0).clip(upper=1.0)
grouped["raw_volume_only"] = 0.50 * grouped["volume_score"]
grouped = grouped.sort_values(
    ["current_count", "high_count"], ascending=False
)

out = root / "submission" / "confidence_diagnostic_20260511.csv"
grouped.to_csv(out, index=False)

print("Week:", week.date())
print("Store/issue groups with at least 2 reports:", len(grouped))
print("Groups with 5+ reports:", int((grouped["current_count"] >= 5).sum()))
print("Groups with any high/critical predictions:", int((grouped["high_count"] > 0).sum()))
print("Diagnostic saved:", out.resolve())
print(grouped.head(15).to_string(index=False))
