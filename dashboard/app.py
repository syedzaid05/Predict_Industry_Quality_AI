"""Predict Industrial Quality AI — Streamlit dashboard entry point.

Standard Streamlit multipage architecture (st.navigation + st.Page).
Run: streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

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

# First-boot model bootstrap (Streamlit Community Cloud: models/ is git-ignored)
with st.spinner("Preparing models (first run trains automatically)..."):
    from ui_common import ensure_models

    if ensure_models():
        st.cache_resource.clear()

pages_dir = _HERE / "pages"

from ui_common import inject_theme

inject_theme()

nav = st.navigation(
    {
        "Operations": [
            st.Page(str(pages_dir / "live_monitoring.py"), title="Real-Time Manual Monitoring", icon=":material/monitoring:", default=True),
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
    st.markdown(
        """
        <div style="letter-spacing:3px;font-weight:800;font-size:1.02rem;color:#ffffff;">
            PREDICT&nbsp;·&nbsp;QUALITY&nbsp;·&nbsp;AI
        </div>
        <div style="color:#aeb6c2;font-size:.78rem;margin-top:4px;letter-spacing:.5px;">
            Predictive Maintenance Suite
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.divider()
    st.caption("Use the sidebar pages to navigate. Works on desktop and phone.")

nav.run()
