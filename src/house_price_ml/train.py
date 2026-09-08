"""End-to-end training, cross-validation, tuning, and final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_validate

from .data import load_dataset, make_train_test_split, split_features_target
from .evaluation import regression_metrics
from .models import get_candidate_models, get_tuning_grid
from .pipeline import build_model_pipeline

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def cross_validate_model(pipeline, X_train, y_train, cv) -> dict[str, float]:
    """Run cross-validation on training data only and return positive error metrics."""
    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "rmse_mean": float(-results["test_rmse"].mean()),
        "rmse_std": float(results["test_rmse"].std()),
        "mae_mean": float(-results["test_mae"].mean()),
        "mae_std": float(results["test_mae"].std()),
        "r2_mean": float(results["test_r2"].mean()),
        "r2_std": float(results["test_r2"].std()),
    }


def train_project(
    data_path: str | Path,
    target_column: str,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    """Train, tune, and evaluate the project without touching the test set until the end."""
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(data_path)
    X, y = split_features_target(dataframe, target_column)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    models = get_candidate_models(random_state=random_state)

    cv_rows = []
    for name, estimator in models.items():
        pipeline = build_model_pipeline(X_train, estimator, dense_output=True)
        metrics = cross_validate_model(pipeline, X_train, y_train, cv)
        cv_rows.append({"model": name, **metrics})

    cv_table = pd.DataFrame(cv_rows).sort_values("rmse_mean")
    cv_table.to_csv(report_dir / "cross_validation_results.csv", index=False)

    best_name = str(cv_table.iloc[0]["model"])
    best_pipeline = build_model_pipeline(
        X_train,
        models[best_name],
        dense_output=True,
    )
    grid = get_tuning_grid().get(best_name, {})

    if grid:
        search = GridSearchCV(
            best_pipeline,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        final_pipeline = search.best_estimator_
        tuning_summary = {
            "selected_model": best_name,
            "best_params": search.best_params_,
            "best_cv_rmse": float(-search.best_score_),
        }
    else:
        final_pipeline = best_pipeline.fit(X_train, y_train)
        tuning_summary = {
            "selected_model": best_name,
            "best_params": {},
            "best_cv_rmse": None,
        }

    y_pred = final_pipeline.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)

    joblib.dump(final_pipeline, model_dir / "final_model.joblib")
    with (report_dir / "tuning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(tuning_summary, file, indent=2)
    with (report_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    return {
        "selected_model": best_name,
        "cv_results": cv_table.to_dict(orient="records"),
        "tuning": tuning_summary,
        "test_metrics": test_metrics,
        "model_path": str(model_dir / "final_model.joblib"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the house-price regression project.")
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output-dir", default=".", help="Project output directory.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = train_project(
        args.data,
        args.target,
        args.output_dir,
        random_state=args.random_state,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
