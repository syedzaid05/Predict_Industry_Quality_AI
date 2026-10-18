"""Predict Industrial Quality AI — Streamlit dashboard entry point.

Standard Streamlit multipage architecture (st.navigation + st.Page).
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="Predict Industrial Quality AI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="auto",
)

pages_dir = _HERE / "pages"

nav = st.navigation(
    {
        "Operations": [
            st.Page(str(pages_dir / "single_prediction.py"), title="Single Prediction", icon=":material/home:", default=True),
            st.Page(str(pages_dir / "live_monitoring.py"), title="Real-Time Manual Monitoring", icon=":material/monitoring:"),
            st.Page(str(pages_dir / "batch_csv.py"), title="Batch CSV Prediction", icon=":material/upload_file:"),
        ],
        "Intelligence": [
            st.Page(str(pages_dir / "health_analytics.py"), title="Machine Health Analytics", icon=":material/insights:"),
            st.Page(str(pages_dir / "explainability.py"), title="Explainability (SHAP)", icon=":material/lightbulb:"),
            st.Page(str(pages_dir / "system_monitoring.py"), title="System Monitoring", icon=":material/troubleshoot:"),
        ],
    },
    position="sidebar",
)

with st.sidebar:
    st.markdown("**Predict Industrial Quality AI**")
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.divider()
    st.caption("Use the sidebar pages to navigate. Works on desktop and phone.")

nav.run()
