"""Batch CSV prediction page."""
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import GRADE_STYLE, get_predictor, hero, note_card, render_batch_results, section_header
from predict_industrial_quality_ai.validation import standardize_columns, validate_dataframe

hero(
    "Batch CSV Prediction",
    "Upload an AI4I-format CSV. Rows are validated, then every machine is scored in one pass.",
    badge="OPERATIONS · BULK SCORING",
)

note_card(
    "<b>How to demo this page:</b> load a sample (1k/10k/20k rows) or your own CSV — any CSV "
    "containing the 5 sensor readings is accepted: custom column names are mapped automatically "
    "(e.g. <b>Air Temp</b>, <b>rpm</b>, <b>Tool wear (min)</b>), Type values may be "
    "L/M/H or low/medium/high (or missing), and comma / semicolon / tab separators all work. "
    "Rows are validated, scored, graded HEALTHY / WARNING / INSPECT / FAILURE RISK, and "
    "exportable for your maintenance plan."
)

SAMPLES = {
    "Small (1k rows)": "ai4i_synthetic_1k.csv",
    "Medium (10k rows)": "ai4i_synthetic_10k.csv",
    "Large (20k rows)": "ai4i_synthetic_20k.csv",
}

uploaded = st.file_uploader("CSV file — any CSV with the 5 sensor readings works (AI4I format, custom headers, comma/semicolon/tab)", type=["csv"])

if not uploaded:
    st.markdown("No file handy? Try a sample dataset from the `data/` folder:")
    cols = st.columns(len(SAMPLES))
    for col, (label, fname) in zip(cols, SAMPLES.items()):
        with col:
            sample_path = _HERE.parent / "data" / fname
            if st.button(f"Load sample: {label}", key=f"sample_{fname}", use_container_width=True):
                if sample_path.exists():
                    st.session_state["sample_csv"] = sample_path.read_bytes()
                    st.session_state["sample_name"] = fname
                    st.session_state.pop("batch", None)
                    st.rerun()
                else:
                    st.error(f"Sample not found: {sample_path}")

if not uploaded and "sample_csv" in st.session_state:
    uploaded = io.BytesIO(st.session_state["sample_csv"])
    uploaded.name = st.session_state["sample_name"]
    st.info(f"Using sample: `{st.session_state['sample_name']}`")

if uploaded and st.button("Run batch prediction", type="primary"):
    try:
        # accept any separator: comma, semicolon, tab …
        raw = pd.read_csv(io.BytesIO(uploaded.read()), sep=None, engine="python",
                          on_bad_lines="skip")
    except Exception:
        raw = pd.read_csv(io.BytesIO(uploaded.read()))
    try:
        raw, col_notes = standardize_columns(raw, "batch upload")
        for n in col_notes:
            st.warning(f"Schema: {n}")
        df, issues = validate_dataframe(raw, "batch upload")
        for issue in issues:
            st.warning(f"Validation: {issue}")

        decisions = get_predictor().predict(df, explain=False)
        results = pd.DataFrame(
            [{k: d[k] for k in ("decision", "failure_probability", "anomaly_score",
                                "agreement", "maintenance_priority")} for d in decisions]
        )
        out = pd.concat([raw.loc[df.index].reset_index(drop=True), results], axis=1)
        st.session_state["batch"] = out

        render_batch_results(out)

        counts = out["decision"].value_counts()
        section_header("Decision distribution", "Fleet-wide mix of graded verdicts")
        order = ["FAILURE_RISK", "WARNING", "INSPECT", "HEALTHY"]
        fig = go.Figure(go.Bar(
            x=[o for o in order if o in counts.index],
            y=[counts[o] for o in order if o in counts.index],
            marker=dict(
                color=[GRADE_STYLE[o]["hue"] for o in order if o in counts.index],
                line=dict(width=0),
            ),
            hovertemplate="%{x}: %{y} machines<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=10), height=340,
            yaxis_title="Machines",
        )
        st.plotly_chart(fig, width="stretch")

        buffer = io.BytesIO()
        out.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download results CSV", buffer, file_name="predictions.csv", type="primary")
    except Exception as exc:
        st.error(f"Batch failed: {exc}")
elif "batch" in st.session_state:
    st.info("Showing the previous batch results.")
    render_batch_results(st.session_state["batch"])
