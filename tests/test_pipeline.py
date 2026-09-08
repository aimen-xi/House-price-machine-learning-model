import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from house_price_ml.data import make_train_test_split, split_features_target
from house_price_ml.pipeline import build_model_pipeline, get_feature_groups


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "size": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
            "rooms": [3, 3, 4, 4, 5, 5, 6, 6],
            "neighborhood": ["A", "A", "B", "B", "C", "C", "A", "D"],
            "target": [100, 120, 140, 160, 180, 200, 220, 240],
        }
    )


def test_feature_target_split() -> None:
    df = sample_dataframe()
    X, y = split_features_target(df, "target")
    assert "target" not in X.columns
    assert len(X) == len(y) == len(df)


def test_feature_groups_are_inferred_from_dtypes() -> None:
    X, _ = split_features_target(sample_dataframe(), "target")
    numeric, categorical = get_feature_groups(X)
    assert numeric == ["size", "rooms"]
    assert categorical == ["neighborhood"]


def test_pipeline_handles_missing_numeric_and_unknown_category() -> None:
    df = sample_dataframe()
    X, y = split_features_target(df, "target")
    X.loc[0, "size"] = np.nan

    X_train, X_test, y_train, _ = make_train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_test = X_test.copy()
    X_test.loc[X_test.index[0], "neighborhood"] = "NEW"

    pipeline = build_model_pipeline(X_train, LinearRegression())
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()
