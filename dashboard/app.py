"""Layer 6 - Streamlit dashboard entrypoint.

Premium-themed, mobile-friendly UI.
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

# ---------------------------------------------------------------- premium theme
st.markdown(
    """<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap" rel="stylesheet">""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
      :root{--gold:#e8b64c;}
      html, body, [data-testid="stAppViewContainer"]{font-family:'Manrope',sans-serif !important;}
      header[data-testid="stHeader"]{background:transparent;}
      [data-testid="stAppViewContainer"]::before{
        content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
        background:
          radial-gradient(800px 420px at 85% -10%, rgba(232,182,76,.10), transparent 60%),
          radial-gradient(700px 380px at -10% 30%, rgba(77,124,254,.09), transparent 60%),
          linear-gradient(165deg,#0b0d12 0%,#10151e 55%,#141a26 100%);
      }
      [data-testid="stAppViewContainer"] > .main{background:transparent;}
      section[data-testid="stSidebar"]{background:rgba(16,20,28,.92);border-right:1px solid rgba(232,182,76,.18);}
      section[data-testid="stSidebar"] *{color:#dfe5ef !important;}
      .block-container{padding-top:1.1rem;max-width:1180px;}
      h1,h2,h3{color:#f6f8fc !important;letter-spacing:-.02em;}
      h1{font-weight:800 !important;}
      div[data-testid="stForm"],div[data-testid="stExpander"],div[data-testid="stDataFrame"],div[data-testid="stPlotlyChart"],div[data-testid="stMetric"]{
        background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.015)) !important;
        border:1px solid rgba(255,255,255,.09) !important;border-radius:16px !important;
      }
      div[data-testid="stForm"]{box-shadow:0 10px 30px rgba(0,0,0,.35);}
      input{background:rgba(255,255,255,.06) !important;border:1px solid rgba(255,255,255,.14) !important;
        border-radius:10px !important;color:#eef2fa !important;font-size:.95rem;}
      input:focus{border-color:var(--gold) !important;box-shadow:0 0 0 3px rgba(232,182,76,.18) !important;}
      label,.stMarkdown p{color:#b9c2d4;}
      button[kind="primary"]{
        background:linear-gradient(135deg,#e8b64c 0%,#c9912f 100%) !important;
        color:#16130a !important;font-weight:800;border:none;border-radius:12px;
        box-shadow:0 6px 18px rgba(232,182,76,.28);transition:.15s;
      }
      button[kind="primary"]:hover{filter:brightness(1.07);transform:translateY(-1px);}
      button[kind="secondary"]{background:rgba(255,255,255,.06) !important;border:1px solid rgba(255,255,255,.14) !important;
        color:#dfe5ef !important;border-radius:12px;}
      div[data-testid="stMetric"]{border-radius:14px;padding:.7rem .9rem .5rem;}
      div[data-testid="stMetricValue"]{color:#fff;font-weight:800;}
      div[data-testid="stMetricLabel"] p{color:#8f9bad;font-size:.8rem;}
      div[data-testid="stAlert"],div[data-testid="stButton"] button{border-radius:12px;}
      div[data-testid="stCaptionContainer"]{color:#8f9bad !important;}
      @media (max-width:640px){.block-container{padding-left:.8rem;padding-right:.8rem;}h1{font-size:1.4rem !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ nav + hero helpers
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
st.divider()

page = st.session_state["page"]


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div style="margin-bottom:-4px;">
          <div style="display:flex;align-items:center;gap:.55rem;">
            <div style="height:26px;width:4px;border-radius:2px;background:linear-gradient(180deg,#e8b64c,#c9912f);"></div>
            <h1 style="margin:0;font-weight:800;">{title}</h1>
          </div>
          <div style="color:#93a0b4;margin:.25rem 0 .2rem .9rem;font-size:.95rem;">{subtitle}</div>
          <div style="height:2px;width:72px;border-radius:2px;margin:.35rem 0 .6rem .9rem;
                      background:linear-gradient(90deg,#e8b64c,transparent);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


SI_UNITS = {
    "Air temperature [K]": "K (kelvin)",
    "Process temperature [K]": "K (kelvin)",
    "Rotational speed [rpm]": "rpm (rev/min)",
    "Torque [Nm]": "Nm (newton·metre)",
    "Tool wear [min]": "min (minutes)",
}
NUM_COLS = list(SI_UNITS)

with st.sidebar:
    st.markdown("### ⚙ Predict Industrial Quality AI")
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.markdown("---")
    st.markdown("""<span style="font-size:.9rem;">On a phone, use the pill buttons at the top of the page to move between sections.</span>""", unsafe_allow_html=True)
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


def sensor_columns(prefix=""):
    """Two-column sensor input grid showing unit + range on every field."""
    c1, c2 = st.columns(2)
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
        container = c1 if i % 2 == 0 else c2
        with container:
            st.caption(
                f"Range {lo:g} – {hi:g} · SI: {SI_UNITS[col_name]}"
            )
            outs[col_name] = st.number_input(
                col_name, min_value=lo, max_value=hi, value=default, key=f"{prefix}{col_name}"
            )
    return outs


def results_table(decisions):
    rows = [
        {k: d[k] for k in ("decision", "failure_probability", "anomaly_score", "agreement", "maintenance_priority")}
        for d in decisions
    ]
    return pd.DataFrame(rows)


# ============================================================ SINGLE PREDICTION
if page == "🏠 Single Prediction":
    hero("Single Machine Prediction", "Sensor readings — validated, scored by the ML engine and explained with SHAP.")
    names = sensor_columns()
    col3 = st.columns(2)[1]
    with col3:
        tp = st.selectbox("Machine type", ["L (Low)", "M (Medium)", "H (High)"])
    cta1, cta2 = st.columns([1, 2])
    with cta1:
        explain = st.checkbox("SHAP explanation", True)
        risk = st.slider("Risk threshold", 0.5, 0.95, 0.7, 0.05)

    if st.button("RUN PREDICTION", type="primary", use_container_width=True):
        try:
            rec = validate_single_record({**names, "Type": tp.split(" ")[0]})
            df, issues = validate_dataframe(pd.DataFrame([rec]), "single input")
            for i in issues:
                st.warning(i)
            pred = get_predictor()
            pred.risk_threshold = risk
            d = pred.predict(df, explain=explain)[0]
            color = {"HEALTHY": "#3ddc84", "WARNING": "#f5a623", "INSPECT": "#f5a623", "FAILURE_RISK": "#ff5252"}[d["decision"]]
            st.markdown(
                f"""
                <div style="padding:1rem 1.2rem;border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02));border:1px solid {color}55;">
                  <div style="display:flex;align-items:center;gap:.7rem;">
                    <div style="height:38px;width:8px;border-radius:4px;background:{color};"></div>
                    <div>
                      <div style="color:#8f9bad;font-size:.8rem;">DECISION</div>
                      <div style="color:{color};font-size:1.5rem;font-weight:800;">{d['decision']}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Failure probability", f"{d['failure_probability']:.3f}")
            m2.metric("Anomaly score", f"{d['anomaly_score']:.3f}")
            m3.metric("Maintenance priority", d["maintenance_priority"])
            m4.metric("Model agreement", d["agreement"].split(":")[0])
            if d["shap"]:
                st.markdown("**Top SHAP drivers** (positive = pushes towards failure)")
                for name, val in d["shap"]:
                    max_abs = max((abs(v) for _, v in d["shap"]), default=1.0)
                    bar_w = int(min(abs(val) / max(max_abs, 1e-9) * 100, 100))
                    st.markdown(
                        f"""<div style="display:flex;align-items:center;gap:.6rem;margin:.15rem 0;">
                              <div style="width:180px;color:#b9c2d4;font-size:.85rem;">{name}</div>
                              <div style="flex:1;background:rgba(255,255,255,.06);border-radius:8px;height:10px;">
                                <div style="width:{bar_w}%;height:10px;border-radius:8px;
                                background:linear-gradient(90deg,{'#ff5252' if val >= 0 else '#3ddc84'},transparent);"></div>
                              </div>
                              <div style="width:80px;color:{'#ff8a8a' if val >= 0 else '#7de8a9'};">{val:+.4f}</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Invalid input: {e}")

# =============================================== REAL-TIME MANUAL MONITORING
elif page == "⏱ Real-Time Monitoring (Manual)":
    hero("Real-Time Monitoring · Manual Entry", "Type each reading as it arrives — scored, logged and drift-checked instantly.")
    if "rt_log" not in st.session_state:
        st.session_state["rt_log"] = pd.DataFrame(
            columns=["time", "decision", "failure_probability", "anomaly_score",
                     *NUM_COLS, "Type", "drift_sigma"]
        )

    with st.form("rt_form", border=True):
        names = sensor_columns(prefix="rt_")
        tp = st.selectbox("Machine type", ["L (Low)", "M (Medium)", "H (High)"], key="rt_tp")
        add_threshold = st.slider("Failure-risk threshold", 0.5, 0.95, 0.7, 0.05, key="rt_thr")
        submitted = st.form_submit_button("ADD READING", type="primary", use_container_width=True)

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
            color = {"HEALTHY": "#3ddc84", "WARNING": "#f5a623", "INSPECT": "#f5a623", "FAILURE_RISK": "#ff5252"}[d["decision"]]
            st.markdown(
                f"""<div style="padding:.9rem 1.1rem;border-radius:16px;background:linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02));border:1px solid {color}55;">
                      <span style="color:#8f9bad;font-size:.8rem;">LATEST READING</span>
                      <div style="color:{color};font-size:1.4rem;font-weight:800;">{d['decision']} — {d['maintenance_priority']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
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
        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(px.line(log, x="time", y="failure_probability", markers=True,
                                    title="Failure probability over readings"), use_container_width=True)
        with pc2:
            st.plotly_chart(px.line(log, x="time", y="Torque [Nm]", markers=True,
                                    title="Torque over readings (drift watch)"), use_container_width=True)
        with st.expander("Full manual-entry log", expanded=False):
            st.dataframe(log.sort_values("time", ascending=False), use_container_width=True)

# ========================================================= BATCH CSV PREDICTION
elif page == "📄 Batch CSV Prediction":
    hero("Batch CSV Prediction", "Upload an AI4I-format file — validation and scoring for every machine at once.")
    up = st.file_uploader("CSV (AI4I 2020 format)", type=["csv"])
    if up and st.button("RUN BATCH", type="primary"):
        raw = pd.read_csv(io.BytesIO(up.read()))
        try:
            df, issues = validate_dataframe(raw, "batch upload")
            for i in issues:
                st.warning(f"Validation: {i}")
            decisions = get_predictor().predict(df, explain=False)
            out = raw.loc[df.index].copy() if len(raw) == len(df) else df.copy()
            out = pd.concat([out.reset_index(drop=True), results_table(decisions)], axis=1)
            st.session_state["batch"] = out
            st.session_state["batch_raw_issues"] = issues
            st.dataframe(out, use_container_width=True)
            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button("DOWNLOAD RESULTS", buf, file_name="predictions.csv", type="primary")
            counts = out["decision"].value_counts()
            st.plotly_chart(px.bar(x=counts.index, y=counts.values, labels={"x": "Decision", "y": "Machines"},
                                   title="Decision distribution"), use_container_width=True)
        except Exception as e:
            st.error(f"Batch failed: {e}")
    elif "batch" in st.session_state:
        st.dataframe(st.session_state["batch"], use_container_width=True)

# ======================================================== MACHINE HEALTH ANALYTICS
elif page == "📊 Machine Health Analytics":
    hero("Machine Health Analytics", "Fleet-level insight on the latest batch: risk mix, weak spots, distributions.")
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
    hero("Explainability · SHAP", "Which signals drive each failure-risk prediction — and by how much.")
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
            "**Reading SHAP:** positive values push toward failure, negative toward healthy. Magnitude = effect strength for that machine."
        )
    except Exception as e:
        st.error(f"Model not available: {e}")

# =================================================================== MONITORING
elif page == "🩺 Monitoring":
    hero("System Monitoring", "Input drift versus training reference · classifier–detector agreement.")
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
