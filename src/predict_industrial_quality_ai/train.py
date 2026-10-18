"""Layer 4 - ML intelligence engine.

Failure classifier: Random Forest or XGBoost (selectable).
Anomaly detector: Isolation Forest (unsupervised, trained on healthy rows).
Outputs are independent and later combined by the decision layer.
"""
from __future__ import annotations

import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import FEATURE_COLS, add_engineered_features, build_pipeline
from .logging_config import get_logger

logger = get_logger(__name__)

TARGET = "Machine failure"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


def make_classifier(model_type: str = "random_forest", random_state: int = 42):
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=10,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def load_ai4i_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def train(
    data_path: str,
    model_type: str = "random_forest",
    artifact_dir: str | None = None,
) -> dict:
    """Train classifier + anomaly detector, log to MLflow, version artifacts."""
    t0 = time.time()
    artifact_dir = artifact_dir or MODEL_DIR
    os.makedirs(artifact_dir, exist_ok=True)

    df = load_ai4i_csv(data_path)
    df = add_engineered_features(df)

    y = df[TARGET].astype(int) if TARGET in df.columns else None
    X = df[["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
            "Torque [Nm]", "Tool wear [min]", "Type"]]

    if y is not None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        X_tr, y_tr, X_te, y_te = X, None, None, None

    pipe = build_pipeline(make_classifier(model_type))
    pipe.fit(X_tr, y_tr)
    logger.info("Trained %s on %d rows", model_type, len(X_tr))

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    train_sample = X_tr if y_tr is None else X_tr[y_tr == 0]
    iso.fit(pipe.named_steps["preprocessor"].transform(train_sample))  # healthy-data latents

    metrics = {"model_type": model_type, "train_rows": len(X_tr)}
    if y_te is not None:
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        metrics.update(
            {
                "test_accuracy": accuracy_score(y_te, pred),
                "test_precision": precision_score(y_te, pred, zero_division=0),
                "test_recall": recall_score(y_te, pred, zero_division=0),
                "test_specificity": tn / max(tn + fp, 1),
                "test_f1": f1_score(y_te, pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_te, proba),
                "test_pr_auc": average_precision_score(y_te, proba),
                "test_tp": int(tp),
                "test_fp": int(fp),
                "test_tn": int(tn),
                "test_fn": int(fn),
                "test_pos_rate": float(y_te.mean()),
            }
        )
        logger.info("Test metrics: %s", metrics)

    version = time.strftime("%Y%m%d_%H%M%S")
    clf_path = os.path.join(artifact_dir, f"classifier_{model_type}_{version}.joblib")
    iso_path = os.path.join(artifact_dir, f"anomaly_detector_{version}.joblib")
    joblib.dump(pipe, clf_path)
    joblib.dump(iso, iso_path)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("predict_industrial_quality_ai")
    with mlflow.start_run(run_name=f"{model_type}_{version}"):
        if y_te is not None:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", len(X_tr))
        signature = infer_signature(X_tr.head(50), pipe.predict_proba(X_tr.head(50))[:, 1])
        mlflow.sklearn.log_model(
            pipe,
            "classifier_pipeline",
            signature=signature,
            skops_trusted_types=[
                "predict_industrial_quality_ai.features.FeatureEngineer",
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        mlflow.sklearn.log_model(iso, "anomaly_detector", skops_trusted_types=["sklearn.tree._tree.Tree"])

    latest_dir = os.path.join(artifact_dir, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(latest_dir, "classifier.joblib"))
    joblib.dump(iso, os.path.join(latest_dir, "anomaly_detector.joblib"))
    joblib.dump({"classifier": clf_path, "anomaly": iso_path, "metrics": metrics},
                os.path.join(latest_dir, "registry.joblib"))

    ref = {
        "mean": X[["Air temperature [K]", "Process temperature [K]",
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].mean().to_dict(),
        "std": X[["Air temperature [K]", "Process temperature [K]",
                  "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]].std().to_dict(),
    }
    joblib.dump(ref, os.path.join(latest_dir, "reference_stats.joblib"))

    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["classifier_path"] = clf_path
    metrics["anomaly_path"] = iso_path
    return metrics


def load_models(artifact_dir: str | None = None):
    artifact_dir = artifact_dir or os.path.join(MODEL_DIR, "latest")
    pipe = joblib.load(os.path.join(artifact_dir, "classifier.joblib"))
    iso = joblib.load(os.path.join(artifact_dir, "anomaly_detector.joblib"))
    return pipe, iso
