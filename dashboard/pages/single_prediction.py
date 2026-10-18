"""Single machine prediction page."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import streamlit as st

from ui_common import (
    get_predictor,
    machine_type_select,
    render_decision,
    sensor_input_grid,
)
from predict_industrial_quality_ai.validation import validate_dataframe, validate_single_record

st.title("Single Prediction")
st.caption("Enter sensor readings, then run the prediction. All inputs are range-checked and explained.")

values = sensor_input_grid()
machine_type = machine_type_select()

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        explain = st.checkbox("Include SHAP explanation", value=True)
    with col2:
        risk_threshold = st.slider("Risk threshold (FAILURE_RISK)", 0.5, 0.95, 0.7, 0.05)

if st.button("Run prediction", type="primary"):
    try:
        record = validate_single_record({**values, "Type": machine_type})
        df, issues = validate_dataframe(pd.DataFrame([record]), "single prediction")
        for issue in issues:
            st.warning(issue)
        predictor = get_predictor()
        predictor.risk_threshold = risk_threshold
        render_decision(predictor.predict(df, explain=explain)[0])
    except Exception as exc:
        st.error(f"Invalid input: {exc}")
