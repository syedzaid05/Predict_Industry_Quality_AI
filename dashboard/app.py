"""Layer 6 - Streamlit dashboard entrypoint.

Pages: Single Prediction | Batch CSV | Machine Health Analytics |
Explainability | Monitoring (data drift / prediction stats).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from predict_industrial_quality_ai.decision import Predictor
from predict_industrial_quality_ai.train import load_models
from predict_industrial_quality_ai.validation import RANGES, validate_dataframe, validate_single_record

st.set_page_config(page_title="Predict Industrial Quality AI", page_icon="🔧", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    pipe, iso = load_models()
    return Predictor(pipe, iso)


@st.cache_data
def get_reference_stats():
    """Reference stats for the monitoring page (stored at training time)."""
    try:
        import joblib, os
        reg = os.path.join("models", "latest", "reference_stats.joblib")
        return joblib.load(reg)
    except Exception:
        return None


NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

PAGES = {
    "🏠 Predict": "🏠 Single Prediction",
    "⏱ Live": "⏱ Real-Time Monitoring (Manual)",
    "📄 Batch": "📄 Batch CSV Prediction",
    "📊 Health": "📊 Machine Health Analytics",
    "🔍 Explain": "🔍 Explainability",
    "🩺 Monitor": "🩺 Monitoring",
}

st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {background: rgba(15,18,25,0.85);}
      [data-testid="stAppViewContainer"] {background: linear-gradient(160deg,#0e1117 0%,#171d29 55%,#1d2433 100%);}
      section[data-testid="stSidebar"] {background:#11151d;}
      .block-container {padding-top: 1.2rem; max-width: 1200px;}
      .nav-btn {
        display:flex; flex-direction:column; align-items:center; gap:2px;
        padding:.55rem .2rem; border-radius:14px; font-weight:600; font-size:.86rem;
        border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.05);
        color:#e6e9f0 !important; text-decoration:none !important; transition:.15s;
      }
      .nav-btn.active, .nav-btn:hover {background:#4d7cfe; border-color:#4d7cfe; color:#fff !important;}
      div[data-testid="stMetricValue"] {color:#f0f4ff;}
      div[data-testid="stMetricLabel"] p {color:#9aa7bd;}
      div[data-testid="stPlotlyChart"] > div {background:rgba(255,255,255,.03); border-radius:14px;}
      .stTabs, .stDataFrame {border-radius:12px;}
      @media (max-width: 640px) {
        .nav-btn {font-size:.72rem; padding:.5rem .1rem;}
        h1 {font-size:1.45rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state["page"] = PAGES["🏠 Predict"]

# Touch-friendly top navigation (works on phones; no sidebar needed)
nav_cols = st.columns(len(PAGES), gap="small")
for (label, target), col in zip(PAGES.items(), nav_cols):
    is_active = st.session_state["page"] == target
    if col.button(label, key=f"nav_{label}", use_container_width=True,
                  type="primary" if is_active else "secondary"):
        st.session_state["page"] = target
        st.rerun()
st.divider()

page = st.session_state["page"]

with st.sidebar:
    st.markdown("### 🔧 Predict Industrial Quality AI")
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.markdown("---")
    st.caption("On a phone, use the pill buttons at the top of the page to move between sections.")
    if len(st.session_state.get("rt_log", [])):
        st.metric("Manual readings logged", len(st.session_state["rt_log"]))


def results_table(decisions):
    rows = [{k: d[k] for k in ("decision", "failure_probability", "anomaly_score", "agreement", "maintenance_priority")} for d in decisions]
    return pd.DataFrame(rows)


if page == "🏠 Single Prediction":
    st.title("🔧 Single Machine Prediction")
    col1, col2 = st.columns(2)
    with col1:
        at = st.number_input("Air temperature (K) — SI: kelvin, 250–400", 250.0, 400.0, 298.1)
        pt = st.number_input("Process temperature (K) — SI: kelvin, 250–420", 250.0, 420.0, 308.6)
        rs = st.number_input("Rotational speed (rpm) — SI: rev/min, 500–3500", 500.0, 3500.0, 1551.0, step=1.0)
    with col2:
        tq = st.number_input("Torque (Nm) — SI: newton·metre, 0–120", 0.0, 120.0, 42.8)
        tw = st.number_input("Tool wear (min) — SI: minutes, 0–350", 0, 350, 228, step=1)
        tp = st.selectbox("Machine type (quality)", ["L (Low)", "M (Medium)", "H (High)"])
    explain = st.checkbox("Include SHAP explanation", True)
    risk = st.slider("Risk threshold (FAILURE_RISK)", 0.5, 0.95, 0.7, 0.05)

    if st.button("Predict", type="primary"):
        try:
            rec = validate_single_record({
                "Air temperature [K]": at, "Process temperature [K]": pt,
                "Rotational speed [rpm]": rs, "Torque [Nm]": tq,
                "Tool wear [min]": tw, "Type": tp.split(" ")[0],
            })
            df, issues = validate_dataframe(pd.DataFrame([rec]), "single input")
            for i in issues:
                st.warning(i)
            pred = get_predictor()
            pred.risk_threshold = risk
            d = pred.predict(df, explain=explain)[0]
            color = {"HEALTHY": "green", "WARNING": "orange", "INSPECT": "orange", "FAILURE_RISK": "red"}[d["decision"]]
            st.markdown(f"## Decision : [{color}]{d['decision']}[/]")
            c1, c2, c3 = st.columns(3)
            c1.metric("Failure probability", f"{d['failure_probability']:.3f}")
            c2.metric("Anomaly score", f"{d['anomaly_score']:.3f}")
            c3.metric("Maintenance priority", d["maintenance_priority"])
            st.caption(f"Model agreement: {d['agreement']}")
            if d["shap"]:
                st.markdown("**Top SHAP drivers** (positive = pushes towards failure)")
                for name, val in d["shap"]:
                    st.markdown(f"`{name}` : {'+' if val >= 0 else ''}{val}")
        except Exception as e:
            st.error(f"Invalid input: {e}")

elif page == "⏱ Real-Time Monitoring (Manual)":
    st.title("⏱ Real-Time Monitoring — Manual Entry")
    st.markdown("Type each sensor reading as it comes in. Every entry is validated, predicted, and appended to a live log with drift checks — no CSV required.")

    if "rt_log" not in st.session_state:
        st.session_state["rt_log"] = pd.DataFrame(
            columns=["time", "decision", "failure_probability", "anomaly_score",
                     "Air temperature [K]", "Process temperature [K]",
                     "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type",
                     "drift_sigma"]
        )

    def num_label(col: str) -> str:
        lo, hi = RANGES[col]
        return f"{col} ({lo:g}–{hi:g})"

    SI_UNITS = {
        "Air temperature [K]": "K (kelvin)",
        "Process temperature [K]": "K (kelvin)",
        "Rotational speed [rpm]": "rpm (rev/min)",
        "Torque [Nm]": "Nm (newton·metre)",
        "Tool wear [min]": "min (minutes)",
    }

    with st.form("rt_form", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption(f"🌡 Air temperature — SI unit: {SI_UNITS['Air temperature [K]']}, range {RANGES['Air temperature [K]'][0]:g}–{RANGES['Air temperature [K]'][1]:g}")
            at = st.number_input("Air temperature (K)", RANGES["Air temperature [K]"][0], RANGES["Air temperature [K]"][1], 298.1, key="rt_at")
            st.caption(f"🌡 Process temperature — SI unit: {SI_UNITS['Process temperature [K]']}, range {RANGES['Process temperature [K]'][0]:g}–{RANGES['Process temperature [K]'][1]:g}")
            pt = st.number_input("Process temperature (K)", RANGES["Process temperature [K]"][0], RANGES["Process temperature [K]"][1], 308.6, key="rt_pt")
        with c2:
            st.caption(f"⚙️ Rotational speed — SI unit: {SI_UNITS['Rotational speed [rpm]']}, range {RANGES['Rotational speed [rpm]'][0]:g}–{RANGES['Rotational speed [rpm]'][1]:g}")
            rs = st.number_input("Rotational speed (rpm)", RANGES["Rotational speed [rpm]"][0], RANGES["Rotational speed [rpm]"][1], 1551.0, step=1.0, key="rt_rs")
            st.caption(f"🌀 Torque — SI unit: {SI_UNITS['Torque [Nm]']}, range {RANGES['Torque [Nm]'][0]:g}–{RANGES['Torque [Nm]'][1]:g}")
            tq = st.number_input("Torque (Nm)", RANGES["Torque [Nm]"][0], RANGES["Torque [Nm]"][1], 42.8, key="rt_tq")
        with c3:
            st.caption(f"🔧 Tool wear — SI unit: {SI_UNITS['Tool wear [min]']}, range {RANGES['Tool wear [min]'][0]:g}–{RANGES['Tool wear [min]'][1]:g}")
            tw = st.number_input("Tool wear (min)", RANGES["Tool wear [min]"][0], RANGES["Tool wear [min]"][1], 0, step=1, key="rt_tw")
            tp = st.selectbox("Machine type", ["L (Low)", "M (Medium)", "H (High)"], help="Quality variant L/M/H", key="rt_tp")
        add_threshold = st.slider("Failure-risk threshold", 0.5, 0.95, 0.7, 0.05, key="rt_thr")
        submitted = st.form_submit_button("➕ Add reading (real-time check)", type="primary")

    if submitted:
        try:
            rec = validate_single_record({
                "Air temperature [K]": at, "Process temperature [K]": pt,
                "Rotational speed [rpm]": rs, "Torque [Nm]": tq,
                "Tool wear [min]": tw, "Type": tp.split(" ")[0],
            })
            df, _ = validate_dataframe(pd.DataFrame([rec]), "manual reading")
            pred = get_predictor()
            pred.risk_threshold = add_threshold
            d = pred.predict(df, explain=True)[0]

            ref = get_reference_stats()
            refstd = ref["std"]["Torque [Nm]"] if ref else 1.0
            drift = abs(tq - (ref["mean"]["Torque [Nm]"] if ref else tq)) / max(refstd, 1e-9)

            st.session_state["rt_log"].loc[len(st.session_state["rt_log"])] = {
                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "decision": d["decision"],
                "failure_probability": d["failure_probability"],
                "anomaly_score": d["anomaly_score"],
                "Air temperature [K]": at, "Process temperature [K]": pt,
                "Rotational speed [rpm]": rs, "Torque [Nm]": tq,
                "Tool wear [min]": tw, "Type": tp.split(" ")[0],
                "drift_sigma": round(drift, 2),
            }
            color = {"HEALTHY": "green", "WARNING": "orange", "INSPECT": "orange", "FAILURE_RISK": "red"}[d["decision"]]
            st.markdown(f"### Latest reading: :{color}[{d['decision']}] — {d['maintenance_priority']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Failure probability", f"{d['failure_probability']:.3f}")
            m2.metric("Anomaly score", f"{d['anomaly_score']:.3f}")
            m3.metric("Agreement", d["agreement"].split(":")[0])
            m4.metric("Torque drift", f"{drift:.2f}σ")
            if d["shap"]:
                st.markdown("**Why:** " + " · ".join(f"`{n}` {v:+.3f}" for n, v in d["shap"][:3]))
        except Exception as e:
            st.error(f"Rejected: {e}")

    log = st.session_state["rt_log"]
    if len(log):
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings", len(log))
        c2.metric("High risk", int((log["decision"] == "FAILURE_RISK").sum()))
        c3.metric("Last state", log.iloc[-1]["decision"])
        st.plotly_chart(px.line(log, x="time", y="failure_probability", markers=True,
                                title="Failure probability over readings"), use_container_width=True)
        st.plotly_chart(px.line(log, x="time", y="Torque [Nm]", markers=True,
                                title="Torque over readings (drift watch)"), use_container_width=True)
        with st.expander("Full manual-entry log"):
            st.dataframe(log.sort_values("time", ascending=False), use_container_width=True)

elif page == "📄 Batch CSV Prediction":
    st.title("📄 Batch CSV Prediction")
    up = st.file_uploader("Upload AI4I-format CSV", type=["csv"])
    if up and st.button("Run batch prediction", type="primary"):
        raw = pd.read_csv(io.BytesIO(up.read()))
        try:
            df, issues = validate_dataframe(raw, "batch upload")
            for i in issues:
                st.warning(f"Validation: {i}")
            decisions = get_predictor().predict(df, explain=False)
            out = raw.loc[df.index].copy() if len(raw) == len(df) else df.copy()
            out = pd.concat([out.reset_index(drop=True), results_table(decisions)], axis=1)
            st.session_state["batch"] = out
            st.dataframe(out, use_container_width=True)
            buf = io.BytesIO(); out.to_csv(buf, index=False); buf.seek(0)
            st.download_button("Download results CSV", buf, file_name="predictions.csv")
            counts = out["decision"].value_counts()
            st.plotly_chart(px.bar(x=counts.index, y=counts.values, labels={"x": "Decision", "y": "Machines"},
                                   title="Decision distribution"), use_container_width=True)
        except Exception as e:
            st.error(f"Batch failed: {e}")
    elif "batch" in st.session_state:
        st.dataframe(st.session_state["batch"], use_container_width=True)

elif page == "📊 Machine Health Analytics":
    st.title("📊 Machine Health Analytics")
    if "batch" not in st.session_state:
        st.info("Run a batch prediction first (Batch CSV page) — analytics are computed on those results.")
    else:
        out = st.session_state["batch"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines", len(out))
        c2.metric("High risk", int((out["decision"] == "FAILURE_RISK").sum()))
        c3.metric("Warnings", int(out["decision"].isin(["WARNING", "INSPECT"]).sum()))
        c4.metric("Healthy", int((out["decision"] == "HEALTHY").sum()))
        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(px.pie(out, names="decision", title="Decision share", hole=0.4),
                            use_container_width=True)
        with colB:
            if "Type" in out:
                st.plotly_chart(px.histogram(out, x="Type", color="decision", barmode="stack",
                                             title="Risk by machine type"), use_container_width=True)
        st.plotly_chart(px.histogram(out, x="failure_probability", nbins=25, color="decision",
                                     title="Failure probability distribution"), use_container_width=True)
        for col in NUM_COLS:
            if col in out:
                st.plotly_chart(px.histogram(out, x=col, color="decision", nbins=30,
                                             title=f"{col} vs decision"), use_container_width=True)

elif page == "🔍 Explainability":
    st.title("🔍 Explainability (SHAP)")
    st.info("Run single/batch predictions with SHAP enabled; this page explains how to read them and shows global feature importance if a model is loaded.")
    try:
        pipe, _ = load_models()
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            names = pipe.named_steps["preprocessor"].get_feature_names_out()
            imp = pd.DataFrame({"feature": [n.replace("num__", "").replace("cat__", "") for n in names],
                                "importance": model.feature_importances_}).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                                   title="Model feature importance"), use_container_width=True)
        st.markdown(
            "**Reading SHAP values:** positive SHAP values push the prediction towards failure, "
            "negative values towards healthy. Magnitude = strength of the effect for that specific machine."
        )
    except Exception as e:
        st.error(f"Model not available: {e}")

elif page == "🩺 Monitoring":
    st.title("🩺 Monitoring")
    st.subheader("Input drift vs training reference")
    ref = get_reference_stats()
    if ref is None:
        st.warning("No reference statistics stored — retrain with the training script to enable drift checks.")
    else:
        if "batch" not in st.session_state:
            st.info("Run a batch prediction first.")
        else:
            out = st.session_state["batch"]
            rows = []
            for col in NUM_COLS:
                if col in out and col in ref["mean"]:
                    cur_mean, cur_std = float(out[col].mean()), float(out[col].std())
                    r_mean, r_std = ref["mean"][col], max(ref["std"][col], 1e-9)
                    drift = abs(cur_mean - r_mean) / r_std
                    rows.append({"feature": col, "train_mean": r_mean, "current_mean": round(cur_mean, 2),
                                 "drift_sigma": round(drift, 2), "status": "DRIFT" if drift > 1.0 else "OK"})
            drift_df = pd.DataFrame(rows)
            st.dataframe(drift_df, use_container_width=True)
            st.plotly_chart(px.bar(drift_df, x="feature", y="drift_sigma", color="status",
                                   title="Input drift (sigma units vs training set)"), use_container_width=True)
    st.subheader("Model agreement / neutrality")
    if "batch" in st.session_state:
        out = st.session_state["batch"]
        st.metric("Classifier ↔ anomaly detector disagreement rate",
                  f"{(out['agreement'].str.startswith('disagree')).mean():.1%}")
    st.caption("Batch predictions are kept in session state; production deployments would log these to MLflow / a warehouse.")
