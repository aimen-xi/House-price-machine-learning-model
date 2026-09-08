"""Preprocessing and model-pipeline construction."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and non-numeric feature names.

    The split is inferred from dtypes so the pipeline does not depend on
    dataset-specific column names.
    """
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features were found in the input dataframe.")

    return numeric_features, categorical_features


def build_preprocessor(
    X: pd.DataFrame,
    *,
    dense_output: bool = True,
) -> ColumnTransformer:
    """Build a leakage-safe preprocessing transformer.

    Numeric columns use median imputation followed by standardization.
    Non-numeric columns use most-frequent imputation followed by one-hot
    encoding. Unknown categories at inference time are ignored.

    ``dense_output=True`` is convenient for the tree models used here. For
    high-cardinality categorical data, sparse output is preferable to reduce
    memory usage.
    """
    numeric_features, categorical_features = get_feature_groups(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0 if dense_output else 1.0,
    )


def build_model_pipeline(
    X: pd.DataFrame,
    model,
    *,
    dense_output: bool = True,
) -> Pipeline:
    """Combine preprocessing and an estimator into one sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, dense_output=dense_output)),
            ("model", model),
        ]
    )
