"""Batch inference utilities and command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict_csv(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a serialized pipeline and generate predictions for a CSV file."""
    model = joblib.load(model_path)
    dataframe = pd.read_csv(input_path)
    predictions = model.predict(dataframe)

    result = dataframe.copy()
    result["prediction"] = predictions

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate house-price predictions from a CSV file.")
    parser.add_argument("--model", required=True, help="Path to final_model.joblib")
    parser.add_argument("--input", required=True, help="CSV containing feature columns")
    parser.add_argument("--output", help="Optional CSV path for predictions")
    args = parser.parse_args()

    result = predict_csv(args.model, args.input, args.output)
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
