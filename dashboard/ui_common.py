"""Shared UI components for all dashboard pages.

Standard Streamlit patterns only:
- cached model loading
- consistent decision cards
- consistent sensor-input block with SI units and allowed ranges
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from predict_industrial_quality_ai.decision import Predictor
from predict_industrial_quality_ai.train import load_models
from predict_industrial_quality_ai.validation import RANGES

# ------------------------------------------------------------------ constants

SENSORS = {
    "Air temperature [K]": {"unit": "K", "default": 298.1},
    "Process temperature [K]": {"unit": "K", "default": 308.6},
    "Rotational speed [rpm]": {"unit": "rpm", "default": 1551.0},
    "Torque [Nm]": {"unit": "Nm", "default": 42.8},
    "Tool wear [min]": {"unit": "min", "default": 0.0},
}
SENSOR_COLS = list(SENSORS)
MACHINE_TYPES = ["L (Low)", "M (Medium)", "H (High)"]

DECISION_COLORS = {
    "HEALTHY": "green",
    "WARNING": "orange",
    "INSPECT": "orange",
    "FAILURE_RISK": "red",
}

# Refined grade palette — saturated jewel tones, fully visible on dark bg.
GRADE_STYLE = {
    "HEALTHY": {
        "hue": "#00e676", "dark": "#00b25c",
        "label": "HEALTHY", "tag": "P4 · NO ACTION REQUIRED",
        "tier": 0,
    },
    "WARNING": {
        "hue": "#ffc400", "dark": "#c79100",
        "label": "WARNING", "tag": "P3 · MONITOR",
        "tier": 1,
    },
    "INSPECT": {
        "hue": "#ff9100", "dark": "#c35f00",
        "label": "INSPECT", "tag": "P2 · INSPECT SOON",
        "tier": 1,
    },
    "FAILURE_RISK": {
        "hue": "#ff1744", "dark": "#b3002b",
        "label": "FAILURE RISK", "tag": "P1 · IMMEDIATE ACTION",
        "tier": 2,
    },
}

_PREMIUM_CSS = """
<style>
/* ================= BLACK & GRAY BLENDED THEME — high-contrast text ================ */
body, .stApp {
    background: linear-gradient(180deg,#060708 0%,#0c0d10 40%,#121417 100%) !important;
    color: #f8fafc !important;
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif !important;
    letter-spacing: .3px;
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 3px rgba(0,0,0,.8);
}
p, span, td, li { color: #eef1f5; }
#MainMenu, footer, header { visibility: hidden; }

/* ---------- metrics: black panel, readable white numbers ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(160deg,#15171b,#0d0e11);
    border: 1px solid #2e3440;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,.55);
}
div[data-testid="stMetric"] label {
    color: #c3cad6 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: .75rem !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,.7);
}
div[data-testid="stMetric"] > div:last-child {
    font-weight: 700; color: #ffffff !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #262b33 !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg,#0f1115,#0a0b0e) !important;
}

/* ---------- sidebar: graphite blend ---------- */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg,#111318 0%,#08090b 100%) !important;
}
[data-testid="stSidebar"] * { color: #e8ecf1 !important; }
[data-testid="stSidebar"] hr { border-color: #2c313a !important; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a {
    color: #eef1f7 !important;
}
[data-testid="stSidebar"] a:hover {
    background: #1b1e24; border-radius: 8px;
}

/* ---------- inputs / buttons / expanders ---------- */
.stNumberInput label, .stSelectbox label, .stSlider label, .stTextInput label {
    color: #dfe4ec !important;
    font-weight: 600;
}
.stButton > button {
    border: 1px solid #3a4048;
    background: linear-gradient(160deg,#1a1c21,#111317);
    color: #ffffff !important;
    font-weight: 700;
    border-radius: 10px;
    transition: all .2s ease;
}
.stButton > button:hover {
    box-shadow: 0 4px 18px rgba(255,255,255,.08);
    border-color: #8a94a6;
    color: #fff !important;
}

/* file uploader: black/gray blend, no white flash */
[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
    background: transparent;
}
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(160deg,#1a1d23,#101216) !important;
    border: 1.5px dashed #4a5058 !important;
    border-radius: 14px;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #dfe4ec !important; fill: #8a94a6;
}
[data-testid="stFileUploaderDropzone"] svg { stroke: #8a94a6; }
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #8a94a6 !important;
    box-shadow: 0 4px 18px rgba(255,255,255,.08);
}
[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] button {
    background: transparent;
    border-color: #3a4048;
    color: #dfe4ec;
}
[data-testid="stFileUploaderUploadedFile"] {
    background: #15171b;
    border: 1px solid #2e3440;
    border-radius: 10px;
    color: #dfe4ec !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(160deg,#3a4048,#20242b) !important;
    border: 1px solid #6b7484 !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: .5px;
}
[data-testid="stExpander"] {
    border: 1px solid #262b33 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #ffffff !important; font-weight: 700;
}
.stDataFrame {
    border: 1px solid #262b33;
    border-radius: 12px;
    overflow: hidden;
}
.stCaption, [data-testid="stCaptionContainer"] { color: #b7c0cc !important; }

/* ---------- section headers ---------- */
.sec-header { display: flex; align-items: center; gap: 12px; margin: 26px 0 6px; }
.sec-bar {
    width: 5px; align-self: stretch; border-radius: 3px;
    background: linear-gradient(180deg,#c9d1dc,#4a5058);
    box-shadow: 0 0 8px rgba(255,255,255,.25);
}
.sec-title {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.25rem; font-weight: 800;
    color: #ffffff; letter-spacing: .4px;
    text-shadow: 0 1px 3px rgba(0,0,0,.8);
}
.sec-sub { color: #c3cad6 !important; font-size: .87rem; margin-bottom: 4px; font-weight: 500; }

/* ---------- hero: black-to-graphite blend ---------- */
.hero {
    background: linear-gradient(120deg,#1c1f26 0%,#0b0c0f 60%);
    border: 1px solid #3a4048;
    border-radius: 18px;
    padding: 20px 28px; margin-bottom: 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,.65);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ""; position: absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,#aab4c2,rgba(170,180,194,0) 70%);
}
.hero-badge {
    display:inline-block; font-size:.68rem; font-weight:800;
    letter-spacing:2.5px; color:#e6ebf2; border:1px solid #5a626e;
    border-radius:999px; padding:3px 12px; margin-bottom:8px;
    background:#191c22;
}
.hero-title {
    font-size:1.75rem; font-weight:800; color:#ffffff; letter-spacing:.5px;
    text-shadow: 0 2px 6px rgba(0,0,0,.9);
}
.hero-caption { color:#cfd6e0 !important; font-size:.9rem; margin-top:5px; font-weight: 500; }

/* ---------- pipeline ribbon: silver nodes ---------- */
.pipe { display:flex; gap:0; margin:6px 0 10px; }
.pipe-step { flex:1; text-align:center; position:relative; padding:0 4px; }
.pipe-dot {
    width:42px; height:42px; border-radius:50%; margin:0 auto 8px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(160deg,#23262e,#101216);
    border:2px solid #8a94a6; color:#ffffff;
    font-size:1.05rem; font-weight:800;
    box-shadow:0 0 12px rgba(255,255,255,.18), 0 6px 14px rgba(0,0,0,.6);
}
.pipe-step:not(:first-child)::before {
    content:""; position:absolute; top:20px; left:-50%; width:100%;
    height:2px; background:linear-gradient(90deg, rgba(138,148,166,0), rgba(138,148,166,.6));
    z-index:-1;
}
.pipe-label { color:#ffffff !important; font-size:.74rem; font-weight:800; letter-spacing:1.2px; }
.pipe-desc { color:#b7c0cc !important; font-size:.68rem; margin-top:3px; line-height:1.4; font-weight: 500; }

/* ---------- info callout: graphite ---------- */
.note {
    border-left:4px solid #6b7484;
    border-radius:0 12px 12px 0;
    background:linear-gradient(90deg,#16181d,#0e1014);
    padding:14px 18px; margin:10px 0;
    color:#dfe4ec !important; font-size:.88rem; line-height:1.6;
    box-shadow: 0 6px 18px rgba(0,0,0,.4);
}
.note b { color:#ffffff; font-weight:700; }
</style>
"""


def inject_theme() -> None:
    """Inject the premium global stylesheet once per page."""
    st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------ UI components

_BODY = "margin:0;background:#08090b;font-family:'Segoe UI',sans-serif;"


def _block(html: str, height: int) -> None:
    """Render HTML in an isolated component — immune to markdown parsing."""
    components.html(
        f"<body style=\"{_BODY}\">{html}</body>",
        height=height, scrolling=False,
    )


def hero(title: str, caption: str, badge: str = "PREDICTIVE MAINTENANCE AI") -> None:
    _block(f"""
    <div style="background:linear-gradient(120deg,#1c1f26 0%,#0b0c0f 60%);
                border:1px solid #3a4048;border-radius:18px;
                border-top:3px solid #aab4c2;
                padding:16px 26px;box-shadow:0 14px 40px rgba(0,0,0,.65);">
        <div style="display:inline-block;font-size:.68rem;font-weight:800;
                    letter-spacing:2.5px;color:#e6ebf2;border:1px solid #5a626e;
                    border-radius:999px;padding:3px 12px;margin-bottom:8px;background:#191c22;">{badge}</div>
        <div style="font-size:1.75rem;font-weight:800;color:#ffffff;letter-spacing:.5px;
                    text-shadow:0 2px 6px rgba(0,0,0,.9);">{title}</div>
        <div style="color:#cfd6e0;font-size:.9rem;margin-top:5px;font-weight:500;">{caption}</div>
    </div>
    """, 160)


def section_header(title: str, subtitle: str | None = None) -> None:
    _block(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:20px 0 2px;">
        <div style="width:5px;align-self:stretch;border-radius:3px;
                    background:linear-gradient(180deg,#c9d1dc,#4a5058);"></div>
        <div>
            <div style="font-size:1.22rem;font-weight:800;color:#ffffff;
                        letter-spacing:.4px;text-shadow:0 1px 3px rgba(0,0,0,.8);">{title}</div>
            {f'<div style="color:#c3cad6;font-size:.87rem;margin-top:2px;font-weight:500;">{subtitle}</div>' if subtitle else ''}
        </div>
    </div>
    """, 96)


def pipeline_ribbon(steps: list[tuple[str, str, str]]) -> None:
    """steps: list of (icon, label, description)."""
    cells = ""
    for icon, label, desc in steps:
        cells += f"""
        <div style="flex:1;text-align:center;position:relative;padding:0 4px;">
            <div style="width:42px;height:42px;border-radius:50%;margin:0 auto 8px;
                        display:flex;align-items:center;justify-content:center;
                        background:linear-gradient(160deg,#23262e,#101216);
                        border:2px solid #8a94a6;color:#ffffff;font-size:1.05rem;
                        font-weight:800;box-shadow:0 0 12px rgba(255,255,255,.18),0 6px 14px rgba(0,0,0,.6);">{icon}</div>
            <div style="color:#ffffff;font-size:.74rem;font-weight:800;letter-spacing:1.2px;">{label}</div>
            <div style="color:#b7c0cc;font-size:.68rem;margin-top:3px;line-height:1.4;font-weight:500;">{desc}</div>
        </div>"""
    _block(f'<div style="display:flex;margin:6px 0 10px;">{cells}</div>', 150)


def note_card(text: str) -> None:
    est_lines = max(2, (len(text) // 150) + text.count("<br>") + 2)
    _block(f"""
    <div style="border-left:4px solid #6b7484;border-radius:0 12px 12px 0;
                background:linear-gradient(90deg,#16181d,#0e1014);
                padding:14px 18px;color:#dfe4ec;font-size:.88rem;line-height:1.6;
                box-shadow:0 6px 18px rgba(0,0,0,.4);">{text}</div>
    """, 52 + 16 * est_lines)

# Repo root: dashboard/ui_common.py -> parents[1]
ROOT = Path(__file__).resolve().parents[1]
MODELS_LATEST = ROOT / "models" / "latest"


# ---------------------------------------------------------------- deployment

def ensure_models() -> bool:
    """Train models on first boot if artifacts are missing (e.g. Streamlit
    Community Cloud, where models/ is git-ignored). Returns True if it trained.
    """
    if (MODELS_LATEST / "classifier.joblib").exists() and (MODELS_LATEST / "anomaly_detector.joblib").exists():
        return False

    from predict_industrial_quality_ai.train import train

    data_path = ROOT / "data" / "ai4i_synthetic.csv"
    if not data_path.exists():
        import sys

        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from generate_data import generate

        generate(path=str(data_path))
    train(str(data_path), model_type="random_forest")
    return True


# ----------------------------------------------------------------- model cache

@st.cache_resource(show_spinner="Loading trained models...")
def get_predictor() -> Predictor:
    pipe, iso = load_models()
    return Predictor(pipe, iso)


@st.cache_data
def get_reference_stats() -> dict | None:
    try:
        import joblib

        return joblib.load(MODELS_LATEST / "reference_stats.joblib")
    except Exception:
        return None


# ------------------------------------------------------------------- widgets

def sensor_input_grid(prefix: str = "") -> dict:
    """Two-column sensor inputs. Each label shows the SI unit and allowed range."""
    values: dict = {}
    cols = st.columns(2)
    for i, (name, meta) in enumerate(SENSORS.items()):
        lo, hi = RANGES[name]
        default = meta["default"]
        if prefix.startswith("rt_") and "rt_log" in st.session_state and len(st.session_state["rt_log"]):
            last = st.session_state["rt_log"].iloc[-1]
            if name in last.index:
                default = float(last[name])
        with cols[i % 2]:
            values[name] = st.number_input(
                f"{name}  ·  unit: {meta['unit']}  ·  range {lo:g}–{hi:g}",
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                key=f"{prefix}{name}",
            )
    return values


def machine_type_select(prefix: str = "") -> str:
    choice = st.selectbox("Machine type (quality variant)", MACHINE_TYPES, key=f"{prefix}machine_type")
    return choice.split(" ")[0]


# ------------------------------------------------------------------ rendering

def _risk_gauge(prob: float, hue: str, label: str) -> str:
    """Full-width SVG speedometer: green→yellow→red arc, glowing needle at prob."""
    prob = max(0.0, min(1.0, float(prob)))
    angle = prob * 180  # 0..180 deg
    needle_x = 100 - 82 * __import__("math").cos(__import__("math").radians(angle))
    needle_y = 95 - 82 * __import__("math").sin(__import__("math").radians(angle))
    return f"""
    <div style="background:linear-gradient(160deg,#16181e,#0c0d10);
                border:1px solid #2e3440;border-radius:18px;
                padding:18px 22px 8px;margin-bottom:8px;
                box-shadow:0 10px 30px rgba(0,0,0,.5);">
      <svg viewBox="0 0 200 118" width="100%" style="display:block;margin:auto;max-width:340px;">
        <defs>
          <linearGradient id="gaugeArc" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"  stop-color="#00e676"/>
            <stop offset="40%" stop-color="#ffc400"/>
            <stop offset="70%" stop-color="#ff9100"/>
            <stop offset="100%" stop-color="#ff1744"/>
          </linearGradient>
        </defs>
        <path d="M 18 95 A 82 82 0 0 1 182 95" fill="none"
              stroke="url(#gaugeArc)" stroke-width="15" stroke-linecap="round"/>
        <circle cx="{needle_x:.1f}" cy="{needle_y:.1f}" r="4.5" fill="{hue}">
          <animate attributeName="r" values="4.5;3.2;4.5" dur="1.6s" repeatCount="indefinite"/>
        </circle>
        <line x1="100" y1="95" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
              stroke="{hue}" stroke-width="3.5" stroke-linecap="round"
              style="filter: drop-shadow(0 0 6px {hue});"/>
        <circle cx="100" cy="95" r="7" fill="#20242e" stroke="{hue}" stroke-width="2"/>
        <text x="100" y="86" text-anchor="middle" fill="{hue}" font-size="21"
              font-weight="700" font-family="Segoe UI">{prob*100:.1f}%</text>
        <text x="18" y="112" text-anchor="middle" fill="#c3cad6" font-size="9" font-weight="700">SAFE</text>
        <text x="182" y="112" text-anchor="middle" fill="#c3cad6" font-size="9" font-weight="700">CRITICAL</text>
      </svg>
      <div style="text-align:center;color:#dfe4ec;font-size:.72rem;font-weight:700;
                  letter-spacing:2px;margin-bottom:2px;">{label}</div>
    </div>
    """


def _led_lamp(on: bool, color: str, label: str) -> str:
    glow = (f"box-shadow:0 0 14px 3px {color};" if on
            else "background:#1c1e24;")
    fill = f"background:{color};" if on else ""
    text_c = "#ffffff" if on else "#9aa3b2"
    return f"""
    <div style="flex:1;text-align:center;padding:10px 4px;border-radius:12px;
                background:linear-gradient(180deg,#14161b,#0d0e12);border:1px solid #262b33;">
      <div style="width:20px;height:20px;border-radius:50%;margin:0 auto 6px;
                  border:2px solid rgba(255,255,255,.35);{fill}{glow}"></div>
      <div style="color:{text_c};font-size:.7rem;font-weight:700;letter-spacing:1.5px;">{label}</div>
    </div>
    """


def render_decision(decision: dict) -> None:
    gx = GRADE_STYLE.get(decision["decision"], GRADE_STYLE["WARNING"])
    hue = gx["hue"]
    active = gx["tier"]

    banner = f"""
    <div style="background:linear-gradient(90deg,{gx['dark']},{hue});
                border-radius:16px;padding:16px 24px;margin-bottom:12px;
                box-shadow:0 10px 32px {gx['dark']}55, 0 0 24px {hue}40;
                border:1px solid {hue}">
      <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:16px;height:16px;border-radius:50%;background:{hue};
                    box-shadow:0 0 16px 4px {hue};"></div>
        <div>
          <div style="color:#0b0d12;font-weight:800;font-size:1.5rem;
                      letter-spacing:2.5px;text-shadow:0 1px 0 rgba(255,255,255,.25);">{gx['label']}</div>
          <div style="color:rgba(0,0,0,.75);font-weight:700;font-size:.78rem;
                      letter-spacing:2px;">{gx['tag']}</div>
        </div>
      </div>
    </div>
    """

    # gauge + LEDs + banner in ONE isolated component (markdown-proof)
    _block(
        _risk_gauge(decision["failure_probability"], hue, "FAILURE PROBABILITY")
        + '<div style="display:flex;gap:8px;margin:0 0 14px;">'
        + _led_lamp(active == 0, GRADE_STYLE["HEALTHY"]["hue"], "HEALTHY")
        + _led_lamp(active == 1, GRADE_STYLE["WARNING"]["hue"], "WARNING")
        + _led_lamp(active == 2, GRADE_STYLE["FAILURE_RISK"]["hue"], "FAILURE RISK")
        + "</div>" + banner,
        480,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Failure probability", f"{decision['failure_probability']:.3f}", border=True)
    c2.metric("Anomaly score", f"{decision['anomaly_score']:.3f}", border=True)
    c3.metric("Model agreement", decision["agreement"].split(":")[0], border=True)
    c4.metric("Anomaly detector", "ALERT" if decision["anomaly_flag"] else "OK",
              delta="ALERT" if decision["anomaly_flag"] else "OK",
              delta_color="inverse", border=True)
    if decision["shap"]:
        with st.expander("Explainability — top SHAP drivers", expanded=True):
            st.caption("Positive values push the prediction toward failure, negative toward healthy.")
            shap_df = pd.DataFrame(decision["shap"], columns=["feature", "SHAP value"])
            st.dataframe(shap_df, hide_index=True, width="stretch")


# Plain-English meaning of every grade — used on the scorecard so a
# non-expert can read the interview demo at a glance.
PLAIN_MEANING = {
    "HEALTHY": "Run as planned",
    "WARNING": "Keep an eye on it",
    "INSPECT": "Needs a checkup soon",
    "FAILURE_RISK": "Act now — stop & repair",
}


def render_scorecard(out: pd.DataFrame) -> None:
    """Standalone, always-visible fleet scorecard: 4 grade tiles + distribution bar."""
    counts = out["decision"].value_counts()
    total = max(len(out), 1)
    tiles = '<div style="display:flex;gap:12px;">'
    for key in ("FAILURE_RISK", "WARNING", "INSPECT", "HEALTHY"):
        g = GRADE_STYLE[key]
        n = int(counts.get(key, 0))
        tiles += (
            f'<div style="flex:1;text-align:center;padding:14px 8px;border-radius:14px;'
            f'background:linear-gradient(160deg,#16181e,#0d0e12);'
            f'border:1px solid #3a4048;border-top:3px solid {g["hue"]};'
            f'box-shadow:0 8px 22px rgba(0,0,0,.55);">'
            f'<div style="color:{g["hue"]};font-size:2rem;font-weight:800;'
            f'text-shadow:0 0 14px {g["hue"]}55;">{n}</div>'
            f'<div style="color:#ffffff;font-size:.7rem;font-weight:800;'
            f'letter-spacing:1.6px;margin-top:4px;">{g["label"]}</div>'
            f'<div style="color:#aeb6c2;font-size:.64rem;margin-top:3px;">{PLAIN_MEANING[key]}</div>'
            f'<div style="color:#aeb6c2;font-size:.64rem;margin-top:2px;">{n / total * 100:.0f}% of fleet</div>'
            f'</div>'
        )
    tiles += "</div>"
    bar = '<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;border:1px solid #4a5058;">'
    for key in ("FAILURE_RISK", "WARNING", "INSPECT", "HEALTHY"):
        n = int(counts.get(key, 0))
        if n:
            bar += f'<div style="width:{n / total * 100:.1f}%;background:{GRADE_STYLE[key]["hue"]};"></div>'
    bar += "</div>"
    _block(
        tiles
        + '<div style="margin:14px 0 4px;font-size:.66rem;color:#aeb6c2;'
        'letter-spacing:1.5px;font-weight:700;">FLEET GRADE MIX &nbsp;—&nbsp; '
        f'{total} machines</div>'
        + bar,
        235,
    )


def render_batch_results(out: pd.DataFrame) -> None:
    render_scorecard(out)
    section_header("Detailed results", "Every machine, scored and graded")
    st.dataframe(out, width="stretch")
