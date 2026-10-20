"""Machine health analytics page (batch-driven fleet view)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plotly.express as px
import streamlit as st

from ui_common import GRADE_STYLE, SENSOR_COLS, hero, note_card, render_batch_results, section_header

hero(
    "Machine Health Analytics",
    "Fleet-level view of the most recent batch: risk mix, weak spots, and feature distributions.",
    badge="INTELLIGENCE · FLEET ANALYTICS",
)

note_card(
    "<b>What to show here:</b> the tiles give the one-line fleet verdict — how many machines "
    "are healthy, on watch, or in danger. The donut shows the overall risk mix, the stacked chart identifies "
    "<b>which machine types</b> concentrate the risk, and the histograms reveal <b>which sensor "
    "regime</b> separates failing machines from healthy ones — the evidence base for maintenance "
    "prioritisation."
)

if "batch" not in st.session_state:
    st.info("No batch results yet. Run a prediction on the Batch CSV page first.")
    st.stop()

out = st.session_state["batch"]
section_header("Overall Machine Risk Overview", "How many machines are healthy, on watch, or in danger")
render_batch_results(out)

section_header("Risk composition", "Where the risk sits across the fleet")
GRADE_ORDER = ["HEALTHY", "WARNING", "INSPECT", "FAILURE_RISK"]
# Fixed bivariate palette (distinct from the verdict colors):
# sky blue → teal → violet → coral
BIVAR_MAP = {
    "HEALTHY": "#5aa9f5",
    "WARNING": "#2fd0b6",
    "INSPECT": "#a78bfa",
    "FAILURE_RISK": "#ff7a90",
}

chart1, chart2 = st.columns(2)
with chart1:
    fig = px.pie(out, names="decision", title="Decision share", hole=0.45,
                 category_orders={"decision": GRADE_ORDER},
                 color="decision", color_discrete_map=BIVAR_MAP)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      textfont=dict(color="#ffffff", size=13))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      legend_font_color="#ffffff", margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")
with chart2:
    if "Type" in out.columns:
        fig = px.histogram(out, x="Type", color="decision", barmode="stack",
                           title="Risk concentrated by machine type",
                           category_orders={"decision": GRADE_ORDER},
                           color_discrete_map=BIVAR_MAP)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", legend_font_color="#ffffff",
                          title_font_color="#ffffff",
                          margin=dict(t=44, b=10), bargap=0.25)
        fig.update_xaxes(tickfont_color="#c3cad6")
        fig.update_yaxes(tickfont_color="#c3cad6")
        st.plotly_chart(fig, width="stretch")

section_header("Failure probability spread")
fig = px.histogram(out, x="failure_probability", nbins=25, color="decision",
                   title="Failure probability distribution by grade",
                   category_orders={"decision": GRADE_ORDER},
                   color_discrete_map=BIVAR_MAP)
fig.add_vline(x=0.7, line_dash="dot", line_color="#ff1744",
              annotation_text="P1 threshold", annotation_font_color="#ff1744")
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", legend_font_color="#ffffff",
                  title_font_color="#ffffff", margin=dict(t=44, b=10))
fig.update_xaxes(tickfont_color="#c3cad6")
fig.update_yaxes(tickfont_color="#c3cad6")
st.plotly_chart(fig, width="stretch")

section_header("Feature distributions by decision",
               "Which sensor regimes separate healthy from risky machines")
for col in SENSOR_COLS:
    if col in out.columns:
        fig = px.histogram(out, x=col, color="decision", nbins=30,
                           title=f"{col} vs decision",
                           category_orders={"decision": GRADE_ORDER},
                           color_discrete_map=BIVAR_MAP)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", legend_font_color="#ffffff",
                          title_font_color="#ffffff", margin=dict(t=44, b=10))
        fig.update_xaxes(tickfont_color="#c3cad6")
        fig.update_yaxes(tickfont_color="#c3cad6")
        st.plotly_chart(fig, width="stretch")
