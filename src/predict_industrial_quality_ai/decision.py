"""Layer 5 - Decision layer & explainability.

Combines the independent classifier and anomaly-detector outputs using
risk thresholds and model-disagreement logic, assigns maintenance
priority, and produces SHAP explanations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

GREY_BAND_LOW = 0.4   # classifier risk below this => low
GREY_BAND_HIGH = 0.7  # classifier risk above this => high


class Predictor:
    def __init__(self, pipeline, iso, risk_threshold: float = GREY_BAND_HIGH):
        self.pipe = pipeline
        self.iso = iso
        self.risk_threshold = risk_threshold

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if hasattr(df, "items"):
            if df.index.name is None and len(df.index) == 0:
                df = pd.DataFrame([df], index=[0])
        df = df.copy()
        df["Type"] = df["Type"].astype(str).str.upper()
        return df

    def predict(self, df: pd.DataFrame, explain: bool = False):
        """Returns list of decision dicts incl. optional SHAP payloads."""
        df = self._prepare(df)
        X_t = self.pipe.named_steps["preprocessor"].transform(df)
        proba = self.pipe.predict_proba(df)[:, 1]
        scores = -self.iso.decision_function(X_t)  # higher => more anomalous
        shap_payloads = []
        if explain:
            import shap

            model = self.pipe.named_steps["model"]
            feat_names = self.pipe.named_steps["preprocessor"].get_feature_names_out()
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_t[:100])
            sv = explainer.shap_values(X_t)
            sv = sv[1] if isinstance(sv, list) else sv
            arr = np.asarray(sv)
            if arr.ndim == 3:
                if self.pipe.named_steps["model"].classes_[-1] == 1:
                    arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr[:, :, -1]
                else:
                    arr = arr[:, :, list(self.pipe.named_steps["model"].classes_).index(1)]
            sv = arr
            for i in range(X_t.shape[0]):
                vals = sv[i]
                top = sorted(zip(feat_names, vals), key=lambda p: abs(p[1]), reverse=True)[:4]
                shap_payloads.append([(str(n).replace("num__", "").replace("cat__", ""), round(float(v), 4)) for n, v in top])

        decisions = []
        for i, (p, s) in enumerate(zip(proba, scores)):
            anomaly_flag = bool(s > 0.0)
            if p < GREY_BAND_LOW and not anomaly_flag:
                state = "HEALTHY"
            elif p >= GREY_BAND_HIGH or (p >= GREY_BAND_LOW and anomaly_flag):
                state = "FAILURE_RISK"
            elif p >= GREY_BAND_LOW and not anomaly_flag:
                state = "WARNING"
            else:
                state = "WARNING"

            agreement = "agree" if (p >= self.risk_threshold) == anomaly_flag or p < GREY_BAND_LOW else "disagree"
            if anomaly_flag and p < GREY_BAND_LOW:
                agreement = "disagree: anomaly only"
                state = "INSPECT"
            elif p >= GREY_BAND_HIGH and not anomaly_flag:
                agreement = "disagree: classifier only"

            priority = {"FAILURE_RISK": "P1 - Immediate", "INSPECT": "P2 - Inspect soon",
                        "WARNING": "P3 - Monitor", "HEALTHY": "P4 - None"}[state]
            decisions.append(
                {
                    "failure_probability": round(float(p), 4),
                    "anomaly_score": round(float(s), 4),
                    "anomaly_flag": anomaly_flag,
                    "decision": state,
                    "agreement": agreement,
                    "maintenance_priority": priority,
                    "shap": shap_payloads[i] if shap_payloads else None,
                }
            )
        logger.info("Predicted %d record(s); states=%s", len(decisions),
                    pd.Series([d["decision"] for d in decisions]).value_counts().to_dict())
        return decisions
