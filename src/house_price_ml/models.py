"""Model definitions used by the training workflow."""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def get_candidate_models(random_state: int = 42) -> dict:
    """Return baseline and candidate regression estimators."""
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=random_state,
        ),
    }


def get_tuning_grid() -> dict:
    """Return a deliberately small grid suitable for a reproducible project."""
    return {
        "ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "random_forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [None, 20],
            "model__min_samples_leaf": [1, 2],
        },
        "hist_gradient_boosting": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [150, 250],
            "model__max_leaf_nodes": [15, 31],
        },
    }
