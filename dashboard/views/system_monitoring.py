"""System monitoring page: input drift and model agreement."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
for p in (str(_HERE.parent / "src"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_common import (
    SENSOR_COLS,
    GRADE_STYLE,
    get_reference_stats,
    hero,
    note_card,
    section_header,
    _block as html_block,
)
from predict_industrial_quality_ai.validation import RANGES

hero(
    "System Monitoring",
    "Watch the model itself: input drift versus training data, and classifier–detector agreement.",
    badge="INTELLIGENCE · MODEL HEALTH",
)

note_card(
    "<b>Why this page matters:</b> a predictive-maintenance model is only trustworthy while "
    "live data looks like its training data. This page answers two questions a reviewer always "
    "asks — has the process drifted since training, and do the two independent models (classifier "
    "and anomaly detector) agree? High drift or high disagreement are the model's own alarms."
)

# ------------------------------------------------------------------ drift

reference = get_reference_stats()

if reference is None:
    st.warning("No reference statistics stored — retrain the model to enable drift checks.")
else:
    section_header(
        "Input drift vs training reference",
        "Each bar shows how far the live data's mean has moved from training, in standard deviations (σ)",
    )

    if "batch" not in st.session_state:
        st.info("Run a batch prediction first, then return here to compare its inputs against training.")
    else:
        out = st.session_state["batch"]
        rows = []
        for col in SENSOR_COLS:
            if col in out.columns and col in reference["mean"]:
                current_mean = float(out[col].mean())
                train_mean = reference["mean"][col]
                train_std = max(reference["std"][col], 1e-9)
                drift_sigma = abs(current_mean - train_mean) / train_std
                rows.append(
                    {
                        "feature": col,
                        "train_mean": round(train_mean, 2),
                        "current_mean": round(current_mean, 2),
                        "drift_sigma": round(drift_sigma, 2),
                        "status": "DRIFT" if drift_sigma > 1.0 else "OK",
                    }
                )
        drift_df = pd.DataFrame(rows)

        # graphical drift bar chart with a colored 1σ cutoff
        fig = go.Figure()
        bar_colors = [
            "#ff1744" if s > 1.0 else ("#ffc400" if s > 0.5 else "#00e676")
            for s in drift_df["drift_sigma"]
        ]
        fig.add_trace(go.Bar(
            y=drift_df["feature"],
            x=drift_df["drift_sigma"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="%{y}<br>%{x:.2f}σ<extra></extra>",
        ))
        fig.add_vline(x=1.0, line_dash="dot", line_color="#ff1744",
                      annotation_text="drift limit (1σ)",
                      annotation_font_color="#ff1744")
        fig.update_layout(
            title="Drift per sensor (σ units against training distribution)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10), height=320,
            xaxis_title="| Δ mean | / σ",
        )
        st.plotly_chart(fig, width="stretch")

        # drift table with colored status
        styled = drift_df.copy()
        styled["status"] = styled["status"].map({"OK": "🟢 OK", "DRIFT": "🔴 DRIFT"})
        st.dataframe(styled, width="stretch", hide_index=True)

        n_drift = (drift_df["status"] == "DRIFT").sum()
        if n_drift:
            st.error(
                f"{n_drift} sensor(s) drifted beyond 1σ — the model is scoring data it was not "
                f"trained on. Recommended action: retrain on recent data before trusting P3/P4 grades."
            )
        else:
            st.success("All sensors within 1σ of training distribution — grades remain trustworthy.")
        note_card(
            "<b>Interpretation:</b> green = means essentially unchanged, amber 0–1σ = worth watching, "
            "red = past 1σ. Each cell also shows train vs current means so a reviewer can "
            "see the direction and size of the shift, not just the flag."
        )

# -------------------------------------------------------------- agreement

section_header(
    "Model agreement",
    "Do the failure classifier and the anomaly detector reach the same conclusion?",
)

if "batch" not in st.session_state:
    st.info("Run a batch prediction first to see model agreement.")
else:
    out = st.session_state["batch"]
    disagreement = out["agreement"].str.startswith("disagree").mean()
    rate = float(disagreement)

    hue = GRADE_STYLE["FAILURE_RISK"]["hue"] if rate > 0.25 else (
        GRADE_STYLE["WARNING"]["hue"] if rate > 0.1 else GRADE_STYLE["HEALTHY"]["hue"])
    html_block(f'''
        <div style="background:linear-gradient(120deg,#14171f,#10131a);
                    border:1px solid rgba(255,255,255,.12);border-radius:16px;
                    padding:20px 26px;margin:8px 0 14px;text-align:center;
                    box-shadow:0 10px 30px rgba(0,0,0,.45);">
          <div style="color:{hue};font-size:2.6rem;font-weight:800;
                      text-shadow:0 0 24px {hue}66;">{rate:.1%}</div>
          <div style="color:#9aa6bd;letter-spacing:2.5px;font-size:.75rem;font-weight:700;
                      margin-top:4px;">CLASSIFIER ↔ ANOMALY DISAGREEMENT</div>
          <div style="height:10px;border-radius:6px;background:rgba(255,255,255,.06);
                      margin:14px auto 0;max-width:480px;overflow:hidden;">
            <div style="height:100%;width:{min(rate / .5 * 100, 100):.1f}%;
                        background:linear-gradient(90deg,{GRADE_STYLE['HEALTHY']['hue']},{hue});
                        border-radius:6px;"></div>
          </div>
          <div style="color:#7a8499;font-size:.68rem;margin-top:6px;">
            green zone &lt; 10% &nbsp;·&nbsp; watch &lt; 25% &nbsp;·&nbsp; red zone ≥ 25%
          </div>
        </div>''')
    if rate > 0.25:
        st.error("High disagreement — the two models conflict on many machines. Inspect flagged ones manually and retrain on fresh labels.")
    elif rate > 0.1:
        st.warning("Moderate disagreement — mostly grey-zone machines. Review them individually.")
    else:
        st.success("Models agree on the fleet — confident verdicts.")
    note_card(
        "<b>Interpretation:</b> two independent signals must meet. The <b>classifier</b> learns "
        "label patterns from historical failures; the <b>Isolation Forest</b> flags readings that "
        "simply look unlike anything trained on. Agreement raises confidence; disagreement means "
        "a machine needs a human either way — better reported than silently averaged out."
    )
