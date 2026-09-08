# House Price Prediction — Reusable Tabular ML Pipeline

A production-oriented learning project for house-price regression, built from an initial linear-regression notebook into a reusable, leakage-safe machine-learning workflow.

The project uses the California housing dataset as the working example. The implementation is intentionally dataset-agnostic: preprocessing is inferred from feature dtypes, the target column is configurable, and training/inference are separated from exploratory analysis.

## What this project demonstrates

- Exploratory data analysis before modeling
- Explicit train/test separation
- Median imputation for missing numeric values
- Most-frequent imputation for missing categorical values
- One-hot encoding with `handle_unknown="ignore"`
- Feature scaling for linear models
- A single sklearn `Pipeline` to prevent preprocessing leakage
- `ColumnTransformer` for mixed numeric/categorical data
- Cross-validation for model comparison
- `GridSearchCV` for hyperparameter tuning
- RMSE as the primary selection metric, with MAE and R² reported
- A final holdout evaluation performed only after model selection
- Serialized inference with `joblib`
- Automated tests and CI
- A clean separation between reusable source code and the exploratory notebook

## Repository structure

```text
House-price-machine-learning-model/
├── data/
│   ├── housing.csv
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── house_price_prediction.ipynb
├── reports/
│   └── .gitkeep
├── src/
│   └── house_price_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── evaluation.py
│       ├── models.py
│       ├── pipeline.py
│       ├── predict.py
│       └── train.py
├── tests/
│   ├── test_data.py
│   └── test_pipeline.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Dataset

The working dataset contains 20,640 observations and 10 columns, with `median_house_value` as the target and `ocean_proximity` as a categorical feature. The exploratory analysis identifies 207 missing values in `total_bedrooms`, strong skew in several count-based variables, and a capped target value at 500001.

The dataset is included for reproducibility. Before redistributing this repository or using the dataset in another project, verify the source dataset's licensing and redistribution terms.

## Modeling workflow

The modeling workflow deliberately follows this order:

```text
Raw data
   │
   ├── EDA / data-quality checks
   │
   ▼
Train / test split
   │
   ├── Training data ──► preprocessing ──► CV ──► model selection ──► tuning
   │
   └── Test data ───────────────────────────────────────────────► final evaluation
```

The test set is not used to select preprocessing parameters, compare candidate models, or tune hyperparameters. All learned preprocessing steps live inside the sklearn pipeline and are fitted only on training folds during cross-validation.

### Candidate models

1. Linear Regression — baseline
2. Ridge Regression — regularized linear model
3. Random Forest Regressor — nonlinear ensemble
4. HistGradientBoosting Regressor — nonlinear boosting model

The candidate with the lowest cross-validated RMSE is selected for hyperparameter tuning. The untouched test set is used only once for the final report.

## Evaluation metrics

| Metric | Role | Interpretation |
|---|---|---|
| RMSE | Primary | Penalizes large prediction errors more strongly |
| MAE | Secondary | Average absolute prediction error |
| R² | Secondary | Proportion of target variance explained by the model |

For model selection, lower RMSE is better. RMSE and MAE are expressed in the same units as the target.

## Setup

Python 3.10+ is supported. Python 3.12 is used in CI.

### 1. Clone the repository

```bash
git clone https://github.com/aimen-xi/House-price-machine-learning-model.git
cd House-price-machine-learning-model
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest -q
```

The test suite covers target/feature separation, dtype-based feature discovery, missing-value handling, unseen categorical values, and missing dataset validation.

## Train the project

From the repository root:

```bash
python -m house_price_ml.train \
  --data data/housing.csv \
  --target median_house_value \
  --output-dir .
```

If the package is not installed in editable mode, run from a shell where `src` is on `PYTHONPATH` or install with the command above.

The training command creates:

```text
models/final_model.joblib
reports/cross_validation_results.csv
reports/tuning_summary.json
reports/test_metrics.json
```

Generated model and report artifacts are intentionally ignored by Git. They are reproducible outputs, not source code.

## Generate predictions

After training, provide a CSV containing the same feature columns used during training:

```bash
python -m house_price_ml.predict \
  --model models/final_model.joblib \
  --input path/to/new_houses.csv \
  --output reports/predictions.csv
```

The saved output contains the original input columns plus a `prediction` column.

## Notebook

`notebooks/house_price_prediction.ipynb` is the human-readable EDA and modeling walkthrough. It is intentionally separate from the reusable implementation under `src/` so future projects can reuse the pipeline without copying notebook-specific code.

The notebook covers:

- dataset inspection
- distributions
- correlations
- categorical inspection
- missing-value and duplicate checks
- train/test splitting
- preprocessing design
- baseline regression
- cross-validation
- model comparison
- hyperparameter tuning
- final holdout evaluation

## Reusing the pipeline for another dataset

The reusable API is designed around three pieces of configuration rather than hard-coded feature names:

```python
from house_price_ml.data import load_dataset, split_features_target
from house_price_ml.pipeline import build_model_pipeline
from sklearn.linear_model import LinearRegression

DATA_PATH = "data/another_dataset.csv"
TARGET_COLUMN = "target"

frame = load_dataset(DATA_PATH)
X, y = split_features_target(frame, TARGET_COLUMN)
model = build_model_pipeline(X, LinearRegression())
```

The preprocessing layer automatically separates numeric and non-numeric features. This means a future tabular project can normally change the dataset path and target name without rewriting the preprocessing logic.

For datasets with very high-cardinality categorical features, prefer sparse one-hot output by setting `dense_output=False` when building the pipeline.

## Engineering decisions

### Why preprocessing lives inside the pipeline

Fitting an imputer, scaler, or encoder on the full dataset before cross-validation would leak information from validation folds into training. Keeping every learned transformation inside the sklearn pipeline ensures each fold fits preprocessing only on its own training portion.

### Why the test set is held back

The test set is the final estimate of generalization. Using it repeatedly during model selection turns it into another training signal and makes the reported score optimistic.

### Why the notebook is not the application

Notebooks are useful for investigation and communication. The actual training and inference logic belongs in importable Python modules so it can be tested, reused, automated, and eventually exposed through an API or application.

## CI

GitHub Actions runs on pushes and pull requests and performs:

1. dependency installation
2. Ruff linting
3. pytest

This gives future changes a repeatable quality gate instead of relying only on manual notebook execution.

## Current scope and next improvements

This repository is intentionally scoped as a strong first end-to-end tabular ML project. A production deployment would add experiment tracking, data/version management, model monitoring, structured logging, API serving, and a formal model registry.

The next logical ML improvements are feature engineering, residual analysis, model explainability, and experiment tracking while keeping the holdout test set untouched until the final comparison.

## License

No project license has been added because the dataset's redistribution terms and the desired software license have not been independently verified. Add an appropriate license after confirming both.
