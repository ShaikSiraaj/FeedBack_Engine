
from pathlib import Path
import argparse
import pandas as pd


def load_data(feedback_path):
    """Load the four CSV files needed by the detector."""

    # Find the folder containing feedback.csv
    data_dir = Path(feedback_path).resolve().parent

    # Load each CSV file into a DataFrame
    feedback = pd.read_csv(
        data_dir / "feedback.csv",
        low_memory=False
    )

    store_master = pd.read_csv(
        data_dir / "store_master.csv",
        low_memory=False
    )

    taxonomy = pd.read_csv(
        data_dir / "issue_taxonomy.csv",
        low_memory=False
    )

    week_calendar = pd.read_csv(
        data_dir / "week_calendar.csv",
        low_memory=False
    )

    # Return all four datasets
    return feedback, store_master, taxonomy, week_calendar


def main():
    parser = argparse.ArgumentParser(
        description="Detect emerging store issues"
    )

    parser.add_argument("--feedback", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    # Load the data
    feedback, stores, taxonomy, calendar = load_data(
        args.feedback
    )

    # Print basic information to check that loading worked
    print("Feedback rows:", len(feedback))
    print("Stores:", len(stores))
    print("Issue types:", len(taxonomy))
    print("Calendar rows:", len(calendar))

    print("\nFeedback columns:")
    print(feedback.columns.tolist())

    print("\nData loaded successfully!")


if __name__ == "__main__":
    main()
