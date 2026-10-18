"""System monitoring page: input drift and model agreement."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import SENSOR_COLS, get_reference_stats

st.title("System Monitoring")
st.caption("Input drift versus the training reference, and classifier–anomaly-detector agreement.")

st.subheader("Input drift vs training reference")
reference = get_reference_stats()

if reference is None:
    st.warning("No reference statistics stored — retrain the model to enable drift checks.")
elif "batch" not in st.session_state:
    st.info("Run a batch prediction first to compare its inputs against the training distribution.")
else:
    out = st.session_state["batch"]
    rows = []
    for col in SENSOR_COLS:
        if col in out.columns and col in reference["mean"]:
            current_mean = float(out[col].mean())
            train_mean = reference["mean"][col]
            train_std = max(reference["std"][col], 1e-9)
            drift_sigma = abs(current_mean - train_mean) / train_std
            rows.append(
                {
                    "feature": col,
                    "train_mean": round(train_mean, 2),
                    "current_mean": round(current_mean, 2),
                    "drift_sigma": round(drift_sigma, 2),
                    "status": "DRIFT" if drift_sigma > 1.0 else "OK",
                }
            )
    drift_df = pd.DataFrame(rows)
    st.dataframe(drift_df, width="stretch")
    st.plotly_chart(
        px.bar(drift_df, x="feature", y="drift_sigma", color="status",
               title="Input drift (sigma units vs training set)"),
        width="stretch",
    )

st.subheader("Model agreement")
if "batch" in st.session_state:
    out = st.session_state["batch"]
    disagreement = out["agreement"].str.startswith("disagree").mean()
    st.metric("Classifier ↔ anomaly-detector disagreement rate", f"{disagreement:.1%}")
    st.caption("High disagreement means the two models see conflicting evidence — inspect those machines manually.")
else:
    st.info("Run a batch prediction first to see model agreement.")
