"""Shared preprocessing and model definitions for car price regression."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "MSRP"
RANDOM_STATE = 42

MODEL_CANDIDATES = [
    {"name": "Ridge", "params": {"alpha": 1.0}},
    {"name": "Ridge", "params": {"alpha": 10.0}},
    {"name": "Ridge", "params": {"alpha": 50.0}},
    {
        "name": "ExtraTrees",
        "params": {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 1.0},
    },
    {
        "name": "ExtraTrees",
        "params": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": 1.0},
    },
    {
        "name": "RandomForest",
        "params": {"n_estimators": 200, "min_samples_leaf": 1, "max_features": 0.8},
    },
    {
        "name": "RandomForest",
        "params": {"n_estimators": 200, "min_samples_leaf": 2, "max_features": 0.8},
    },
]


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, int]:
    """Load the source CSV, remove unusable targets and exact duplicate rows."""
    frame = pd.read_csv(data_path)
    frame.columns = frame.columns.str.strip()
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column {TARGET!r} in {data_path}.")

    original_rows = len(frame)
    frame = frame.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    rows_removed = original_rows - len(frame)
    features = frame.drop(columns=TARGET)
    target = pd.to_numeric(frame[TARGET], errors="raise")
    if (target < 0).any():
        raise ValueError("MSRP must be non-negative to use the log1p target transform.")
    return features, target, rows_removed


def make_regressor(model_name: str, params: dict) -> object:
    """Create a fresh estimator so each candidate gets an independent fit."""
    if model_name == "Ridge":
        return Ridge(**params)
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    raise ValueError(f"Unknown model name: {model_name}")


def build_pipeline(features: pd.DataFrame, model_name: str, params: dict):
    """Build imputation, encoding, and log-target regression as one pipeline."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    regressor = Pipeline(
        [("preprocessor", preprocessor), ("model", make_regressor(model_name, params))]
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )