"""Data loading and train/test splitting utilities."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate that it is not empty."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")
    return dataframe


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate a dataframe into features and target."""
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if dataframe[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")

    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible holdout test set."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
