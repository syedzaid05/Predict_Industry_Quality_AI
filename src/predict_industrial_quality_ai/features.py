"""Layer 3 - Feature engineering & training-fitted preprocessing pipeline."""
from __future__ import annotations

import math

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLS = ["Type"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

ENGINEERED_COLS = [
    "Temp diff [K]",
    "Power [W]",
    "Wear x Torque",
]
ENGINEERED_FORMULAS = {
    "Temp diff [K]": "Process temp - Air temp",
    "Power [W]": "Torque * 2*pi * Rotational speed / 60",
    "Wear x Torque": "Tool wear * Torque",
}


def add_engineered_features(df):
    """Derived features from AI4I domain knowledge. Returns new DataFrame."""
    df = df.copy()
    df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    omega = 2 * math.pi * df["Rotational speed [rpm]"] / 60.0
    df["Power [W]"] = df["Torque [Nm]"] * omega
    df["Wear x Torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Adds domain-derived features inside the pipeline (train-fitted)."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        out = add_engineered_features(df)
        self.output_cols_ = [c for c in out.columns]
        self.feature_names_in_ = list(df.columns)
        return self

    def transform(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        out = add_engineered_features(df)
        return out[self.output_cols_]

    def get_feature_names_out(self, input_features=None):
        return self.output_cols_


def build_preprocessor() -> ColumnTransformer:
    """Fits during training; re-applied at inference (train-fitted)."""
    numeric_pipe = Pipeline([("engineer", FeatureEngineer()), ("scale", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
