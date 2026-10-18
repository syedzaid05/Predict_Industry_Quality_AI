"""Layer 6 - Streamlit dashboard entrypoint.

Native dark theme, minimal custom CSS (robust on desktop + mobile).
Pages: Single Prediction | Real-Time Manual Monitoring | Batch CSV |
Machine Health Analytics | Explainability | Monitoring.
"""
import io
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from predict_industrial_quality_ai.decision import Predictor
from predict_industrial_quality_ai.train import load_models
from predict_industrial_quality_ai.validation import (
    RANGES,
    validate_dataframe,
    validate_single_record,
)

st.set_page_config(page_title="Predict Industrial Quality AI", page_icon="⚙", layout="wide")

# Minimal, safe styling only (no widget-internal overrides).
st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; max-width: 1180px;}
      div[data-testid="stMetric"] {background: #ffffff0d; border: 1px solid #ffffff18; border-radius: 12px; padding: 0.6rem 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ navigation
PAGES = {
    "Predict": "🏠 Single Prediction",
    "Live": "⏱ Real-Time Monitoring (Manual)",
    "Batch": "📄 Batch CSV Prediction",
    "Health": "📊 Machine Health Analytics",
    "Explain": "🔍 Explainability",
    "Monitor": "🩺 Monitoring",
}

if "page" not in st.session_state:
    st.session_state["page"] = PAGES["Predict"]

nav_cols = st.columns(len(PAGES), gap="small")
for (label, target), col in zip(PAGES.items(), nav_cols):
    is_active = st.session_state["page"] == target
    if col.button(
        label,
        key=f"nav_{label}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):
        st.session_state["page"] = target
        st.rerun()

page = st.session_state["page"]

with st.sidebar:
    st.title("⚙ Predict IQ-AI")
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.divider()
    st.caption("Use the pill buttons at the top to switch pages — they work best on phones.")
    if "rt_log" in st.session_state:
        st.metric("Manual readings logged", len(st.session_state["rt_log"]))


@st.cache_resource(show_spinner="Loading models...")
def get_predictor():
    pipe, iso = load_models()
    return Predictor(pipe, iso)


@st.cache_data
def get_reference_stats():
    try:
        import joblib

        return joblib.load("models/latest/reference_stats.joblib")
    except Exception:
        return None


SI_UNITS = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "min",
}
NUM_COLS = list(SI_UNITS)


def sensor_columns(prefix=""):
    """Two-column sensor grid; every field shows its SI unit + allowed range."""
    cols = st.columns(2)
    order = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    defaults = {"Air temperature [K]": 298.1, "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551.0, "Torque [Nm]": 42.8, "Tool wear [min]": 0.0}
    outs = {}
    for i, col_name in enumerate(order):
        lo, hi = RANGES[col_name]
        default = defaults[col_name]
        if prefix.startswith("rt_") and "rt_log" in st.session_state and len(st.session_state["rt_log"]):
            last = st.session_state["rt_log"].iloc[-1]
            if col_name in last:
                default = float(last[col_name])
        with cols[i % 2]:
            outs[col_name] = st.number_input(
                f"{col_name} ({SI_UNITS[col_name]})  ·  {lo:g}–{hi:g}",
                min_value=lo, max_value=hi, value=default, key=f"{prefix}{col_name}",
            )
    return outs


def show_decision(d: dict):
    st.success(f"Decision: {d['decision']}  ·  {d['maintenance_priority']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Failure probability", f"{d['failure_probability']:.3f}")
    m2.metric("Anomaly score", f"{d['anomaly_score']:.3f}")
    m3.metric("Model agreement", d["agreement"].split(":")[0])
    m4.metric("Anomaly detected?", "Yes" if d["anomaly_flag"] else "No")
    if d["shap"]:
        with st.expander("Why? — top SHAP drivers (positive = pushes towards failure)", expanded=True):
            for name, val in d["shap"]:
                st.markdown(f"- `{name}` — **{val:+.4f}**")


def results_table(decisions):
    rows = [
        {k: d[k] for k in ("decision", "failure_probability", "anomaly_score", "agreement", "maintenance_priority")}
        for d in decisions
    ]
    return pd.DataFrame(rows)


# ============================================================ SINGLE PREDICTION
if page == "🏠 Single Prediction":
    st.title("Single Machine Prediction")
    st.caption("Enter sensor readings — validated, scored by the ML engine and explained with SHAP.")
    names = sensor_columns()
    tp = st.selectbox("Machine type", ["L (Low)", "M (Medium)", "H (High)"])
    explain = st.checkbox("Include SHAP explanation", True)
    risk = st.slider("Risk threshold (FAILURE_RISK)", 0.5, 0.95, 0.7, 0.05)

    if st.button("Run prediction", type="primary", use_container_width=True):
        try:
            rec = validate_single_record({**names, "Type": tp.split(" ")[0]})
            df, issues = validate_dataframe(pd.DataFrame([rec]), "single input")
            for i in issues:
                st.warning(i)
            pred = get_predictor()
            pred.risk_threshold = risk
            show_decision(pred.predict(df, explain=explain)[0])
        except Exception as e:
            st.error(f"Invalid input: {e}")

# =============================================== REAL-TIME MANUAL MONITORING
elif page == "⏱ Real-Time Monitoring (Manual)":
    st.title("Real-Time Monitoring — Manual Entry")
    st.caption("Type each reading as it arrives. It is scored, logged and drift-checked instantly.")

    if "rt_log" not in st.session_state:
        st.session_state["rt_log"] = pd.DataFrame(
            columns=["time", "decision", "failure_probability", "anomaly_score",
                     *NUM_COLS, "Type", "drift_sigma"]
        )

    with st.form("rt_form", border=True):
        names = sensor_columns(prefix="rt_")
        tp = st.selectbox("Machine type", ["L (Low)", "M (Medium)", "H (High)"], key="rt_tp")
        add_threshold = st.slider("Failure-risk threshold", 0.5, 0.95, 0.7, 0.05, key="rt_thr")
        submitted = st.form_submit_button("Add reading", type="primary", use_container_width=True)

    if submitted:
        try:
            rec = validate_single_record({**names, "Type": tp.split(" ")[0]})
            df, _ = validate_dataframe(pd.DataFrame([rec]), "manual reading")
            pred = get_predictor()
            pred.risk_threshold = add_threshold
            d = pred.predict(df, explain=True)[0]

            ref = get_reference_stats()
            refstd = ref["std"]["Torque [Nm]"] if ref else 1.0
            drift = abs(names["Torque [Nm]"] - (ref["mean"]["Torque [Nm]"] if ref else names["Torque [Nm]"])) / max(refstd, 1e-9)

            st.session_state["rt_log"].loc[len(st.session_state["rt_log"])] = {
                "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "decision": d["decision"],
                "failure_probability": d["failure_probability"],
                "anomaly_score": d["anomaly_score"],
                **names, "Type": tp.split(" ")[0],
                "drift_sigma": round(drift, 2),
            }
            show_decision(d)
        except Exception as e:
            st.error(f"Rejected: {e}")

    log = st.session_state["rt_log"]
    if len(log):
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings", len(log))
        c2.metric("High risk", int((log["decision"] == "FAILURE_RISK").sum()))
        c3.metric("Last state", log.iloc[-1]["decision"])
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(px.line(log, x="time", y="failure_probability", markers=True,
                                    title="Failure probability over readings"), use_container_width=True)
        with pc2:
            st.plotly_chart(px.line(log, x="time", y="Torque [Nm]", markers=True,
                                    title="Torque over readings (drift watch)"), use_container_width=True)
        with st.expander("Full manual-entry log"):
            st.dataframe(log.sort_values("time", ascending=False), use_container_width=True)

# ========================================================= BATCH CSV PREDICTION
elif page == "📄 Batch CSV Prediction":
    st.title("Batch CSV Prediction")
    st.caption("Upload an AI4I-format file — validation and scoring for every machine at once.")
    up = st.file_uploader("CSV (AI4I 2020 format)", type=["csv"])
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
            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("Download results CSV", buf, file_name="predictions.csv", type="primary")
            counts = out["decision"].value_counts()
            st.plotly_chart(px.bar(x=counts.index, y=counts.values, labels={"x": "Decision", "y": "Machines"},
                                   title="Decision distribution"), use_container_width=True)
        except Exception as e:
            st.error(f"Batch failed: {e}")
    elif "batch" in st.session_state:
        st.dataframe(st.session_state["batch"], use_container_width=True)

# ======================================================== MACHINE HEALTH ANALYTICS
elif page == "📊 Machine Health Analytics":
    st.title("Machine Health Analytics")
    st.caption("Fleet-level insight on the latest batch: risk mix, weak spots and distributions.")
    if "batch" not in st.session_state:
        st.info("Run a batch prediction first — analytics are computed on those results.")
    else:
        out = st.session_state["batch"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Machines", len(out))
        c2.metric("High risk", int((out["decision"] == "FAILURE_RISK").sum()))
        c3.metric("Warnings", int(out["decision"].isin(["WARNING", "INSPECT"]).sum()))
        c4.metric("Healthy", int((out["decision"] == "HEALTHY").sum()))
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(px.pie(out, names="decision", title="Decision share", hole=0.45), use_container_width=True)
        with pc2:
            if "Type" in out:
                st.plotly_chart(px.histogram(out, x="Type", color="decision", barmode="stack",
                                             title="Risk by machine type"), use_container_width=True)
        st.plotly_chart(px.histogram(out, x="failure_probability", nbins=25, color="decision",
                                     title="Failure probability distribution"), use_container_width=True)
        for col in NUM_COLS:
            if col in out:
                st.plotly_chart(px.histogram(out, x=col, color="decision", nbins=30,
                                             title=f"{col} vs decision"), use_container_width=True)

# ============================================================== EXPLAINABILITY
elif page == "🔍 Explainability":
    st.title("Explainability (SHAP)")
    st.caption("Which signals drive each failure-risk prediction — and by how much.")
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
            "**Reading SHAP:** positive values push the prediction towards failure, negative values towards healthy. "
            "Magnitude = strength of the effect for that specific machine."
        )
    except Exception as e:
        st.error(f"Model not available: {e}")

# =================================================================== MONITORING
elif page == "🩺 Monitoring":
    st.title("System Monitoring")
    st.caption("Input drift versus training reference · classifier–detector agreement.")
    st.subheader("Input drift vs training reference")
    ref = get_reference_stats()
    if ref is None:
        st.warning("No reference statistics stored — retrain to enable drift checks.")
    elif "batch" not in st.session_state:
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
    st.subheader("Model agreement")
    if "batch" in st.session_state:
        out = st.session_state["batch"]
        st.metric("Classifier ↔ anomaly-detector disagreement rate",
                  f"{(out['agreement'].str.startswith('disagree')).mean():.1%}")
    st.caption("Production deployments would log these stats to MLflow / a warehouse.")
