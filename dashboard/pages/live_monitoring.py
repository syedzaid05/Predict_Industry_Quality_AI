"""Real-time manual monitoring page: type readings one by one, live log."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import (
    SENSOR_COLS,
    get_predictor,
    get_reference_stats,
    machine_type_select,
    render_decision,
    sensor_input_grid,
)
from predict_industrial_quality_ai.validation import validate_dataframe, validate_single_record

st.title("Real-Time Manual Monitoring")
st.caption("Type each sensor reading as it arrives. Every entry is validated, scored, logged, and drift-checked.")

if "rt_log" not in st.session_state:
    st.session_state["rt_log"] = pd.DataFrame(
        columns=["time", "decision", "failure_probability", "anomaly_score",
                 *SENSOR_COLS, "Type", "drift_sigma"]
    )

with st.form("manual_reading_form", border=True):
    values = sensor_input_grid(prefix="rt_")
    machine_type = machine_type_select(prefix="rt_")
    risk_threshold = st.slider("Failure-risk threshold", 0.5, 0.95, 0.7, 0.05, key="rt_threshold")
    submitted = st.form_submit_button("Add reading", type="primary")

if submitted:
    try:
        record = validate_single_record({**values, "Type": machine_type})
        df, _ = validate_dataframe(pd.DataFrame([record]), "manual reading")
        predictor = get_predictor()
        predictor.risk_threshold = risk_threshold
        decision = predictor.predict(df, explain=True)[0]

        ref = get_reference_stats()
        torque = values["Torque [Nm]"]
        drift = 0.0
        if ref:
            drift = abs(torque - ref["mean"]["Torque [Nm]"]) / max(ref["std"]["Torque [Nm]"], 1e-9)

        st.session_state["rt_log"].loc[len(st.session_state["rt_log"])] = {
            "time": pd.Timestamp.now().strftime("%H:%M:%S"),
            "decision": decision["decision"],
            "failure_probability": decision["failure_probability"],
            "anomaly_score": decision["anomaly_score"],
            **values,
            "Type": machine_type,
            "drift_sigma": round(float(drift), 2),
        }
        render_decision(decision)
    except Exception as exc:
        st.error(f"Rejected: {exc}")

log = st.session_state["rt_log"]
if len(log) == 0:
    st.info("No readings yet — add the first one above.")
else:
    st.divider()
    st.subheader("Live log")
    c1, c2, c3 = st.columns(3)
    c1.metric("Readings", len(log))
    c2.metric("High risk", int((log["decision"] == "FAILURE_RISK").sum()))
    c3.metric("Latest state", log.iloc[-1]["decision"])

    chart1, chart2 = st.columns(2)
    with chart1:
        st.plotly_chart(
            px.line(log, x="time", y="failure_probability", markers=True,
                    title="Failure probability over readings"),
            width="stretch",
        )
    with chart2:
        st.plotly_chart(
            px.line(log, x="time", y="Torque [Nm]", markers=True,
                    title="Torque over readings (drift watch)"),
            width="stretch",
        )
    with st.expander("Full log"):
        st.dataframe(log.sort_values("time", ascending=False), width="stretch")
