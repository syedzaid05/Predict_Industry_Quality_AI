"""Shared UI components for all dashboard pages.

Standard Streamlit patterns only:
- cached model loading
- consistent decision cards
- consistent sensor-input block with SI units and allowed ranges
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from predict_industrial_quality_ai.decision import Predictor
from predict_industrial_quality_ai.train import load_models
from predict_industrial_quality_ai.validation import RANGES

# ------------------------------------------------------------------ constants

SENSORS = {
    "Air temperature [K]": {"unit": "K", "default": 298.1},
    "Process temperature [K]": {"unit": "K", "default": 308.6},
    "Rotational speed [rpm]": {"unit": "rpm", "default": 1551.0},
    "Torque [Nm]": {"unit": "Nm", "default": 42.8},
    "Tool wear [min]": {"unit": "min", "default": 0.0},
}
SENSOR_COLS = list(SENSORS)
MACHINE_TYPES = ["L (Low)", "M (Medium)", "H (High)"]

DECISION_COLORS = {
    "HEALTHY": "green",
    "WARNING": "orange",
    "INSPECT": "orange",
    "FAILURE_RISK": "red",
}

# Repo root: dashboard/ui_common.py -> parents[1]
ROOT = Path(__file__).resolve().parents[1]
MODELS_LATEST = ROOT / "models" / "latest"


# ---------------------------------------------------------------- deployment

def ensure_models() -> bool:
    """Train models on first boot if artifacts are missing (e.g. Streamlit
    Community Cloud, where models/ is git-ignored). Returns True if it trained.
    """
    if (MODELS_LATEST / "classifier.joblib").exists() and (MODELS_LATEST / "anomaly_detector.joblib").exists():
        return False

    from predict_industrial_quality_ai.train import train

    data_path = ROOT / "data" / "ai4i_synthetic.csv"
    if not data_path.exists():
        import sys

        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from generate_data import generate

        generate(path=str(data_path))
    train(str(data_path), model_type="random_forest")
    return True


# ----------------------------------------------------------------- model cache

@st.cache_resource(show_spinner="Loading trained models...")
def get_predictor() -> Predictor:
    pipe, iso = load_models()
    return Predictor(pipe, iso)


@st.cache_data
def get_reference_stats() -> dict | None:
    try:
        import joblib

        return joblib.load(MODELS_LATEST / "reference_stats.joblib")
    except Exception:
        return None


# ------------------------------------------------------------------- widgets

def sensor_input_grid(prefix: str = "") -> dict:
    """Two-column sensor inputs. Each label shows the SI unit and allowed range."""
    values: dict = {}
    cols = st.columns(2)
    for i, (name, meta) in enumerate(SENSORS.items()):
        lo, hi = RANGES[name]
        default = meta["default"]
        if prefix.startswith("rt_") and "rt_log" in st.session_state and len(st.session_state["rt_log"]):
            last = st.session_state["rt_log"].iloc[-1]
            if name in last.index:
                default = float(last[name])
        with cols[i % 2]:
            values[name] = st.number_input(
                f"{name}  ·  unit: {meta['unit']}  ·  range {lo:g}–{hi:g}",
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                key=f"{prefix}{name}",
            )
    return values


def machine_type_select(prefix: str = "") -> str:
    choice = st.selectbox("Machine type (quality variant)", MACHINE_TYPES, key=f"{prefix}machine_type")
    return choice.split(" ")[0]


# ------------------------------------------------------------------ rendering

def render_decision(decision: dict) -> None:
    color = DECISION_COLORS.get(decision["decision"], "gray")
    st.subheader(f"Decision: :{color}[{decision['decision']}]  ·  {decision['maintenance_priority']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure probability", f"{decision['failure_probability']:.3f}")
    c2.metric("Anomaly score", f"{decision['anomaly_score']:.3f}")
    c3.metric("Model agreement", decision["agreement"].split(":")[0])
    c4.metric("Anomaly detector", "ALERT" if decision["anomaly_flag"] else "OK")
    if decision["shap"]:
        with st.expander("Explainability — top SHAP drivers", expanded=True):
            st.caption("Positive values push the prediction toward failure, negative toward healthy.")
            shap_df = pd.DataFrame(decision["shap"], columns=["feature", "SHAP value"])
            st.dataframe(shap_df, hide_index=True, width="stretch")


def render_batch_results(out: pd.DataFrame) -> None:
    st.subheader("Results")
    st.dataframe(out, width="stretch")
    counts = out["decision"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Machines", len(out))
    c2.metric("Failure risk", int(counts.get("FAILURE_RISK", 0)))
    c3.metric("Warnings", int(counts.get("WARNING", 0) + counts.get("INSPECT", 0)))
    c4.metric("Healthy", int(counts.get("HEALTHY", 0)))
