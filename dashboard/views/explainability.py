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

from ui_common import GRADE_STYLE, get_predictor, hero, note_card, section_header

hero(
    "Explainability (SHAP)",
    "Which signals drive each failure-risk prediction — and by how much.",
    badge="INTELLIGENCE · TRANSPARENCY",
)

note_card(
    "<b>How to read SHAP values</b> (the model is not a black box): "
    "a <b>positive</b> value pushes the prediction toward <b>failure</b>; a <b>negative</b> value "
    "pushes toward <b>healthy</b>; the <b>magnitude</b> is the strength of that sensor's effect "
    "for the specific machine. Two views are available: the <b>global</b> importance below, and "
    "<b>per-machine</b> drivers on the prediction pages."
)

section_header("Global feature importance", "Model-level: which sensors the forest relies on most")

# Fixed palette for the final demo — matches the app's verdict grading
# (green → amber → orange → red, brightest/warmest = most relied upon).
_SHAP_PALETTE = [
    GRADE_STYLE["HEALTHY"]["hue"],
    GRADE_STYLE["WARNING"]["hue"],
    GRADE_STYLE["INSPECT"]["hue"],
    GRADE_STYLE["FAILURE_RISK"]["hue"],
]

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
        fig = px.bar(importance, x="importance", y="feature", orientation="h",
                     title="Model feature importance (global)",
                     color="importance", color_continuous_scale=_SHAP_PALETTE)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=10),
                          coloraxis_colorbar=dict(tickfont=dict(color="#c3cad6")),
                          title_font_color="#ffffff")
        st.plotly_chart(fig, width="stretch")
    st.caption("Per-machine SHAP drivers are shown on the Real-Time Monitoring tab after each reading.")
except Exception as exc:
    st.error(f"Model not available: {exc}")
