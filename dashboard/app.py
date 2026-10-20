"""Predict Industrial Quality AI — Streamlit dashboard entry point.

Single-page, tabbed architecture (mobile-friendly, like a workspace app).
All sections live on one page: switch with the tab bar, no page reloads.
Run: streamlit run dashboard/app.py
"""
import os
import runpy
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
    initial_sidebar_state="collapsed",
)

# First-boot model bootstrap (Streamlit Community Cloud: models/ is git-ignored)
with st.spinner("Preparing models (first run trains automatically)..."):
    from ui_common import ensure_models

    if ensure_models():
        st.cache_resource.clear()

from ui_common import inject_theme

inject_theme()

with st.sidebar:
    st.markdown(
        "<div style=\"letter-spacing:3px;font-weight:800;font-size:1.02rem;color:#ffffff;\">"
        "PREDICT&nbsp;&middot;&nbsp;QUALITY&nbsp;&middot;&nbsp;AI</div>"
        "<div style=\"color:#aeb6c2;font-size:.78rem;margin-top:4px;letter-spacing:.5px;\">"
        "Predictive Maintenance Suite</div>",
        unsafe_allow_html=True,
    )
    st.caption("Random Forest / XGBoost · Isolation Forest · SHAP")
    st.divider()
    st.caption("Switch sections with the tabs at the top. Everything is on a single page — works smoothly on desktop and phone.")
    if st.button("Use sample data info"):
        st.caption("Sample CSVs live in the data/ folder; load them from the Batch CSV tab.")

# ---------------------------------------------------------------------------
# One page, five workspace tabs (only the active tab runs, so reruns stay fast)
# ---------------------------------------------------------------------------

VIEWS = [
    ("Real-Time Monitoring", "live_monitoring.py", ":material/monitoring:"),
    ("Batch CSV", "batch_csv.py", ":material/upload_file:"),
    ("Health Analytics", "health_analytics.py", ":material/insights:"),
    ("Explainability", "explainability.py", ":material/lightbulb:"),
    ("System Monitor", "system_monitoring.py", ":material/troubleshoot:"),
]

tabs = st.tabs([v[0] for v in VIEWS])

for tab, (label, fname, _icon) in zip(tabs, VIEWS):
    with tab:
        if tab:  # truthy only when this tab is the active one
            runpy.run_path(str(_HERE / "views" / fname), run_name="__tab__")

st.markdown("---")
st.caption("Predict Industrial Quality AI · Random Forest / XGBoost + Isolation Forest + SHAP · all data stays local.")
