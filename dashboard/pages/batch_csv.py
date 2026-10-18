"""Batch CSV prediction page."""
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_predictor, render_batch_results
from predict_industrial_quality_ai.validation import validate_dataframe

st.title("Batch CSV Prediction")
st.caption("Upload an AI4I-format CSV. Rows are validated, then every machine is scored in one pass.")

uploaded = st.file_uploader("CSV file (AI4I 2020 format)", type=["csv"])

if uploaded and st.button("Run batch prediction", type="primary"):
    raw = pd.read_csv(io.BytesIO(uploaded.read()))
    try:
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
        st.plotly_chart(
            px.bar(x=counts.index, y=counts.values,
                   labels={"x": "Decision", "y": "Machines"},
                   title="Decision distribution"),
            width="stretch",
        )

        buffer = io.BytesIO()
        out.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download results CSV", buffer, file_name="predictions.csv", type="primary")
    except Exception as exc:
        st.error(f"Batch failed: {exc}")
elif "batch" in st.session_state:
    st.info("Showing the previous batch results.")
    render_batch_results(st.session_state["batch"])
