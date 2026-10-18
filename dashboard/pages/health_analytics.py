"""Machine health analytics page (batch-driven fleet view)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plotly.express as px
import streamlit as st

from ui_common import SENSOR_COLS, render_batch_results

st.title("Machine Health Analytics")
st.caption("Fleet-level view of the most recent batch: risk mix, weak spots, and feature distributions.")

if "batch" not in st.session_state:
    st.info("No batch results yet. Run a prediction on the Batch CSV page first.")
    st.stop()

out = st.session_state["batch"]
render_batch_results(out)

chart1, chart2 = st.columns(2)
with chart1:
    st.plotly_chart(
        px.pie(out, names="decision", title="Decision share", hole=0.45),
        width="stretch",
    )
with chart2:
    if "Type" in out.columns:
        st.plotly_chart(
            px.histogram(out, x="Type", color="decision", barmode="stack",
                         title="Risk by machine type"),
            width="stretch",
        )

st.plotly_chart(
    px.histogram(out, x="failure_probability", nbins=25, color="decision",
                 title="Failure probability distribution"),
    width="stretch",
)

st.subheader("Feature distributions by decision")
for col in SENSOR_COLS:
    if col in out.columns:
        st.plotly_chart(
            px.histogram(out, x=col, color="decision", nbins=30,
                         title=f"{col} vs decision"),
            width="stretch",
        )
