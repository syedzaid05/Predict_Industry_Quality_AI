"""Real-time manual monitoring page: type readings one by one, live log."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import (
    SENSOR_COLS,
    GRADE_STYLE,
    get_predictor,
    get_reference_stats,
    hero,
    machine_type_select,
    note_card,
    pipeline_ribbon,
    render_decision,
    section_header,
    sensor_input_grid,
)
from predict_industrial_quality_ai.validation import validate_dataframe, validate_single_record

hero(
    "Real-Time Monitoring",
    "Score every sensor reading the moment it arrives: validated → scored → logged → drift-checked.",
    badge="OPERATIONS · LIVE TRIAGE",
)

pipeline_ribbon([
    ("⚙", "1 · INPUT", "Engineer types the latest sensor snapshot"),
    ("✔", "2 · VALIDATE", "Range + type checks reject bad data early"),
    ("🧠", "3 · SCORE", "RF classifier + Isolation Forest vote on risk"),
    ("📋", "4 · DECIDE", "Priority P1–P4 + driver explanation (SHAP)"),
])

note_card(
    "<b>How to read this page:</b> enter one machine's live sensor values below. "
    "The app returns a color-graded verdict — <b style='color:#00e676'>HEALTHY</b> (run normally), "
    "<b style='color:#ffc400'>WARNING</b> (monitor), <b style='color:#ff9100'>INSPECT</b> (check soon) or "
    "<b style='color:#ff1744'>FAILURE RISK</b> (act now, P1) — plus a gauge, a traffic-light panel, "
    "and the top sensor drivers behind the verdict. Every verdict is appended to the live log with a "
    "drift check against training conditions."
)

section_header("Sensor snapshot", "Latest readings from the machine (SI units, validated ranges)")

if "rt_log" not in st.session_state:
    st.session_state["rt_log"] = pd.DataFrame(
        columns=["time", "decision", "failure_probability", "anomaly_score",
                 *SENSOR_COLS, "Type", "drift_sigma"]
    )

with st.form("manual_reading_form", border=True):
    values = sensor_input_grid(prefix="rt_")
    machine_type = machine_type_select(prefix="rt_")
    c1, c2 = st.columns(2)
    with c1:
        risk_threshold = st.slider(
            "Failure-risk threshold", 0.5, 0.95, 0.7, 0.05,
            key="rt_threshold",
            help="Probability above which a reading is graded FAILURE_RISK.",
        )
    with c2:
        explain = st.checkbox(
            "Include SHAP explanation", value=True, key="rt_explain",
            help="Adds the top per-sensor drivers to the verdict card. "
                 "Turn off on slow connections for a snappier response.",
        )
    submitted = st.form_submit_button("▶  Score this reading", type="primary")

if submitted:
    try:
        record = validate_single_record({**values, "Type": machine_type})
        df, _ = validate_dataframe(pd.DataFrame([record]), "manual reading")
        predictor = get_predictor()
        predictor.risk_threshold = risk_threshold
        decision = predictor.predict(df, explain=explain)[0]

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
        section_header("Verdict", "Graded decision for the latest reading")
        render_decision(decision)
    except Exception as exc:
        st.error(f"Rejected: {exc}")

log = st.session_state["rt_log"]
if len(log) == 0:
    st.info("Add a reading above to start the live log.")
else:
    st.divider()

    # -------- trend charts with colored decision markers
    section_header("Trend watch", "Failure probability & lowest-risk indicator drift over the session")
    state_color = {k: GRADE_STYLE[k]["hue"] for k in GRADE_STYLE}

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=log["time"], y=log["failure_probability"], mode="lines+markers",
        line=dict(color="#d4af37", width=2, shape="spline"),
        marker=dict(
            size=11,
            color=[state_color.get(d, "#888") for d in log["decision"]],
            line=dict(width=1.5, color="#0b0d12"),
        ),
        customdata=log["decision"],
        hovertemplate="%{x}<br>probability %{y:.3f}<br>verdict %{customdata}<extra></extra>",
    ))
    fig1.add_hline(y=0.7, line_dash="dot", line_color="#ff1744",
                   annotation_text="P1 threshold", annotation_font_color="#ff1744")
    fig1.add_hline(y=0.4, line_dash="dot", line_color="#ffc400",
                   annotation_text="grey band", annotation_font_color="#ffc400")
    fig1.update_layout(
        title="Failure probability per reading (dot color = verdict)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=10), height=320,
        yaxis=dict(range=[0, 1]),
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=log["time"], y=log["drift_sigma"], mode="lines+markers", fill="tozeroy",
        line=dict(color="#7fd1ff", width=2, shape="spline"),
        fillcolor="rgba(127,209,255,.12)",
        hovertemplate="%{x}<br>drift %{y:.2f}σ<extra></extra>",
    ))
    fig2.add_hline(y=1.0, line_dash="dot", line_color="#ff9100",
                   annotation_text="1σ drift line", annotation_font_color="#ff9100")
    fig2.update_layout(
        title="Torque drift from training conditions (σ units)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=10), height=320,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig1, width="stretch")
    with c2:
        st.plotly_chart(fig2, width="stretch")

    note_card(
        "<b>For review:</b> the left chart shows how fast the machine is approaching the red zone; "
        "dot colors match the verdict grading. The right chart measures how far today's torque "
        "has moved from the conditions the model was trained on — sustained drift (bars past "
        "1σ) is the earliest signal that the model needs retraining."
    )

    with st.expander("Full session log"):
        st.dataframe(log.sort_values("time", ascending=False), width="stretch")
