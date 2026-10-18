"""Explainability (SHAP) page."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_common import get_predictor

st.title("Explainability (SHAP)")
st.caption("Which signals drive each failure-risk prediction — and by how much.")

st.markdown(
    """
**How to read SHAP values**

- **Positive** value: pushes the prediction toward *failure*.
- **Negative** value: pushes the prediction toward *healthy*.
- **Magnitude**: strength of that feature's effect for the specific machine.

Global view (model-level feature importance):
"""
)

try:
    predictor = get_predictor()
    model = predictor.pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = predictor.pipe.named_steps["preprocessor"].get_feature_names_out()
        importance = (
            pd.DataFrame(
                {
                    "feature": [n.replace("num__", "").replace("cat__", "") for n in names],
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance")
        )
        st.plotly_chart(
            px.bar(importance, x="importance", y="feature", orientation="h",
                   title="Model feature importance (global)"),
            width="stretch",
        )
    st.caption("Per-machine SHAP drivers are shown on the prediction pages (Single Prediction and Real-Time Manual Monitoring).")
except Exception as exc:
    st.error(f"Model not available: {exc}")
