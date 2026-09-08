"""Regression evaluation helpers."""

from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate the project's primary regression metrics."""
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validation_summary(cv_results: dict[str, Any]) -> pd.DataFrame:
    """Convert sklearn cross-validation output into a compact results table."""
    rows = []
    for split_name, values in cv_results.items():
        if not split_name.startswith("test_"):
            continue
        metric = split_name.removeprefix("test_")
        values = values.astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )
    return pd.DataFrame(rows)
