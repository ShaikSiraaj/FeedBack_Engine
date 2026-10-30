"""Write the canonical detector re-run command."""

import argparse
from pathlib import Path

DETECTOR_COMMAND = (
    "uv run --project FULL_CODE/pipeline python FULL_CODE/detect.py "
    "--feedback {feedback} --as-of {asof} --out {out}"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"{DETECTOR_COMMAND}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
