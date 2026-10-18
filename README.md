# Predict Industrial Quality AI

An end-to-end, production-shaped **predictive maintenance** system built on the
**AI4I 2020** dataset schema. Given five machine sensor readings and a quality
variant, it predicts the probability of machine failure using two *independent*
models — a supervised **failure classifier** (Random Forest or XGBoost) and an
unsupervised **anomaly detector** (Isolation Forest) — then fuses their outputs
in a **decision layer** that assigns a maintenance state, priority, and a
SHAP-based explanation. A Streamlit dashboard provides single predictions,
real-time manual monitoring, batch scoring, fleet analytics, explainability,
and drift monitoring.

---

## Table of contents

1. [Key features](#1-key-features)
2. [System architecture](#2-system-architecture)
3. [Installation](#3-installation)
4. [Quickstart](#4-quickstart)
5. [The data](#5-the-data)
6. [Layer-by-layer deep dive](#6-layer-by-layer-deep-dive)
7. [Dashboard user guide](#7-dashboard-user-guide)
8. [Training, MLflow and model versioning](#8-training-mlflow-and-model-versioning)
9. [Logging](#9-logging)
10. [Testing](#10-testing)
11. [Project structure](#11-project-structure)
12. [Configuration reference](#12-configuration-reference)
13. [Deployment](#13-deployment)
14. [Troubleshooting FAQ](#14-troubleshooting-faq)
15. [Roadmap](#15-roadmap)

---

## 1. Key features

- **Three data entry paths**: manual typing (real-time), CSV upload (batch),
  and programmatic API (`Predictor.predict()`).
- **Strict data validation** — schema, missing values, numeric coercion, unit
  range checks, duplicate detection, and hard rejection of unusable inputs.
- **Training-fitted preprocessing** — categorical one-hot encoding and
  numeric scaling live *inside* the sklearn `Pipeline`, so inference can never
  leak or diverge from training.
- **Two independent ML models** combined only at the decision layer:
  - Failure classifier: `RandomForestClassifier` or `XGBClassifier` (selectable).
  - Anomaly detector: `IsolationForest` trained exclusively on *healthy* rows.
- **Decision layer** with a risk threshold, a grey band, model-disagreement
  handling, and P1–P4 maintenance priorities.
- **SHAP explainability** (TreeExplainer) for every prediction.
- **Streamlit dashboard** — 6 pages, standard `st.navigation` multipage
  architecture, dark theme, phone-friendly sidebar navigation.
- **Engineering fundamentals** — pytest suite (14 tests), MLflow experiment
  tracking (SQLite backend), timestamped model versioning, file + stdout
  logging.

---

## 2. System architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1 — Data sources                                                  │
│   AI4I 2020 CSV  ·  manual sensor typing  ·  batch CSV upload           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 2 — Data validation  (validation.py)                              │
│   schema check · type coercion · missing values · range checks ·        │
│   duplicate removal · invalid input rejection                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 3 — Feature engineering  (features.py)                            │
│   domain features (ΔT, power, wear×torque)                              │
│   StandardScaler + OneHotEncoder — fitted on training data only         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 4 — ML intelligence engine  (train.py)                            │
│   Failure classifier: RandomForest / XGBoost                            │
│   Anomaly detector:  IsolationForest (healthy rows only)                │
│   → independent probabilities, combined later                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 5 — Decision & explainability  (decision.py)                      │
│   risk threshold · grey band · disagreement states · P1–P4 priority ·   │
│   SHAP top-drivers per machine                                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 6 — Streamlit dashboard  (dashboard/)                             │
│   Single Prediction · Real-Time Manual Monitoring · Batch CSV ·         │
│   Health Analytics · Explainability · System Monitoring                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 7 — Engineering & deployment                                      │
│   pytest · MLflow (SQLite) · logging · model versioning · git ·         │
│   Streamlit Community Cloud ready                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Installation

Requires **Python 3.10+** (developed and tested on 3.13).

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

Packages installed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`,
`numba` (SHAP dependency), `joblib`, `streamlit`, `mlflow`, `pytest`, `plotly`.

---

## 4. Quickstart

```powershell
# 1) Data — either use the real AI4I 2020 CSV, or generate a synthetic stand-in
python scripts/generate_data.py            # → data/ai4i_synthetic.csv (5000 rows)

# 2) Train the models (logs to MLflow, writes versioned artifacts + models/latest/)
python train_cli.py --data data/ai4i_synthetic.csv --model random_forest
#    ...or:            --model xgboost

# 3) Run the test suite
python -m pytest tests -q

# 4) Launch the dashboard
streamlit run dashboard/app.py
```

### Model metrics (20 % hold-out, `data/ai4i_synthetic.csv`, 5,000 rows)

Both classifiers trained on 4,000 rows (15.4 % positive class), evaluated on
1,000 hold-out rows (154 failures, 846 healthy), decision threshold 0.5.
Every metric below is computed and logged by `train.py` and pushed to MLflow
under the `test_*` keys.

| Metric | Definition | Random Forest | XGBoost |
|---|---|---|---|
| **Accuracy** | (TP+TN) / all | **0.955** | 0.948 |
| **Precision** | TP / (TP+FP) | **0.991** | 0.932 |
| **Recall (sensitivity)** | TP / (TP+FN) | 0.714 | 0.714 |
| **Specificity** | TN / (TN+FP) | **0.999** | 0.991 |
| **F1** | harmonic mean of precision & recall | **0.830** | 0.809 |
| **ROC-AUC** | ranking quality across thresholds | **0.856** | 0.853 |
| **PR-AUC** | precision-recall area (imbalanced data) | **0.808** | 0.803 |

**Confusion matrices** (rows = actual, cols = predicted):

| Random Forest | Predicted healthy | Predicted failure |
|---|---|---|
| Actual healthy | TN = **845** | FP = 1 |
| Actual failure | FN = 44 | TP = 110 |

| XGBoost | Predicted healthy | Predicted failure |
|---|---|---|
| Actual healthy | TN = 838 | FP = 8 |
| Actual failure | FN = 44 | TP = 110 |

Reading of these numbers:

- **Random Forest is the default** because it achieves near-perfect precision
  (0.991): when it raises a failure flag, it is almost certainly real — ideal
  for maintenance teams that must trust every alarm.
- Both models catch **110 of 154** real failures (71.4 % recall); the missed
  cases are largely ambiguous mid-band machines, which the **Isolation
  Forest anomaly detector** helps catch via the INSPECT state in the decision
  layer.
- Class imbalance (15.4 % failures) is handled by RF's
  `class_weight="balanced"` and XGB's `scale_pos_weight=10`; PR-AUC is
  therefore the honest headline ranking metric here, not accuracy.

**Anomaly detector (Isolation Forest):**

| Property | Value |
|---|---|
| Training rows | 3,382 healthy-only rows (preprocessed latents) |
| Contamination | 0.05 (5 % expected anomaly rate on healthy training data) |
| Role | independent second opinion; raises INSPECT when the classifier misses anomalies |
| Calibration | `decision_function` sign at 0 → `anomaly_flag` |

> Numbers come from the synthetic dataset (`--seed 42`). With the real UCI
> AI4I 2020 CSV, retrain and compare runs side-by-side in the MLflow UI.

---

## 5. The data

### AI4I 2020 format

The system consumes the
[UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
schema. The 5 sensor columns + `Type` are the model inputs; `Machine failure`
is the binary target used during training only.

| Column | Type | SI unit | Allowed range | Meaning |
|---|---|---|---|---|
| `Air temperature [K]` | float | kelvin | 250 – 400 | Ambient air temperature of the machine cell |
| `Process temperature [K]` | float | kelvin | 250 – 420 | Process fluid temperature |
| `Rotational speed [rpm]` | float | rev/min | 500 – 3500 | Spindle speed |
| `Torque [Nm]` | float | newton·metre | 0 – 120 | Spindle torque |
| `Tool wear [min]` | int | minutes | 0 – 350 | Accumulated tool wear time |
| `Type` | str | — | `L` / `M` / `H` | Low / Medium / High quality variant |
| `Machine failure` | int | — | 0 / 1 | **Target** (training CSVs only) |

Optional columns present in the real dataset (`UDI`, `Product ID`, `TWF`,
`HDF`, `PWF`, `OSF`, `RNF`) are ignored by the pipeline.

Ranges are enforced in **two places**: the dashboard labels every input with
its unit and allowed range, and `validation.RANGES` rejects out-of-range
values at the API layer — you cannot bypass them by typing, uploading, or
calling the code directly.

### Getting the real dataset

1. Download `ai4i2020.csv` from the UCI link above.
2. Place it anywhere (e.g. `data/ai4i2020.csv`) and pass it to the trainer:

```powershell
python train_cli.py --data data/ai4i2020.csv --model xgboost
```

### Synthetic generator

`scripts/generate_data.py` fabricates a schema-compatible dataset with the
same correlated failure physics (over-torque, thermal-gap, power, wear
failure modes plus label noise), useful for demos and CI:

```powershell
python scripts/generate_data.py --n 20000 --path data/big.csv
```

---

## 6. Layer-by-layer deep dive

### Layer 2 — Data validation (`src/.../validation.py`)

Two entry points:

- `validate_dataframe(df)` — for CSV uploads / batch frames. Returns
  `(clean_df, issues)`. In order it performs:
  1. **Schema check** — all required columns present, else `ValidationError`.
  2. **Duplicate removal** — exact duplicate rows dropped, reported.
  3. **Type coercion** — `pd.to_numeric(errors="coerce")` on sensor columns;
     anything non-numeric becomes missing.
  4. **Categorical cleanup** — strips/uppercases `Type`; values outside
     {L, M, H} drop the row.
  5. **Range checks** — values outside `RANGES` drop the row.
  6. **Empty result** — if nothing survives, `ValidationError` is raised.

- `validate_single_record(record)` — for manual typing. Raises
  `ValidationError` with a human-readable message on any violation
  (missing field, non-numeric, bad type, out-of-range).

### Layer 3 — Feature engineering (`src/.../features.py`)

Three domain features are derived inside the pipeline
(`FeatureEngineer` transformer, sklearn-safe):

| Feature | Formula | Rationale |
|---|---|---|
| `Temp diff [K]` | `Process temp − Air temp` | Thermal-gap anomalies (HDF mode) |
| `Power [W]` | `Torque × 2π × rpm / 60` | Mechanical power load (PWF mode) |
| `Wear x Torque` | `Tool wear × Torque` | Combined degradation stress (TWF/OSF modes) |

The full preprocessor is a single `ColumnTransformer`:

- **numeric branch**: `FeatureEngineer` → `StandardScaler`
- **categorical branch**: `OneHotEncoder(handle_unknown="ignore")` on `Type`

It is attached to the model inside one `sklearn.Pipeline`, so it is **fitted
only on training data** and re-applied unchanged at inference. Unknown
category values at inference (e.g. a typo `Type=Z`) are silently ignored
rather than crashing.

### Layer 4 — ML intelligence engine (`src/.../train.py`)

- **Failure classifier** (supervised, trained on all rows with
  stratified 80/20 split):
  - `random_forest` — 300 trees, `class_weight="balanced"`, `min_samples_leaf=2`.
  - `xgboost` — 300 trees, depth 6, lr 0.08, `scale_pos_weight=10`,
    subsample/colsample 0.9.
- **Anomaly detector** (unsupervised) — `IsolationForest(200, contamination=0.05)`
  fitted on the **preprocessed healthy rows only** (`y == 0`). Its
  `decision_function` is negated so higher score ⇒ more anomalous.
- The two models never share gradients or errors; their outputs meet for the
  first time in the decision layer.

Training reports the full metric suite on the hold-out set — accuracy,
precision, recall, specificity, F1, ROC-AUC, PR-AUC and the confusion-matrix
counts (TP/FP/TN/FN) — all stored under `test_*` keys in the trainer output,
in `models/latest/registry.joblib`, and logged to MLflow.

### Layer 5 — Decision layer & explainability (`src/.../decision.py`)

Given classifier probability `p` and anomaly score `s`:

1. `anomaly_flag = (s > 0)`.
2. State machine (grey band `0.4 ≤ p < 0.7`):

| Condition | Decision | Meaning |
|---|---|---|
| `p < 0.4` and no anomaly | **HEALTHY** | Both models calm |
| `0.4 ≤ p < 0.7` and no anomaly | **WARNING** | Classifier uneasy — monitor |
| anomaly flag and `p < 0.4` | **INSPECT** | Detector caught something the classifier missed |
| `p ≥ 0.7`, or (`p ≥ 0.4` and anomaly flag) | **FAILURE_RISK** | Both models or strong classifier signal |

3. **Model agreement** is recorded as `agree`, `disagree: anomaly only`, or
   `disagree: classifier only` — surfaced on every result and aggregated on
   the System Monitoring page.
4. **Maintenance priority** mapping:

| Decision | Priority |
|---|---|
| FAILURE_RISK | P1 — Immediate |
| INSPECT | P2 — Inspect soon |
| WARNING | P3 — Monitor |
| HEALTHY | P4 — None |

5. **SHAP**: `shap.TreeExplainer` on the trained classifier; the top-4
   absolute drivers are returned per machine as `(feature, value)` pairs —
   positive pushes toward failure, negative toward healthy.

The risk threshold on the dashboard sliders (0.50–0.95) adjusts the
FAILURE_RISK boundary; the grey band bounds are constants in `decision.py`.

### Layers 1 & 6 — Dashboard (`dashboard/`)

See the [Dashboard user guide](#7-dashboard-user-guide) below.

### Layer 7 — Engineering

See [Testing](#10-testing), [MLflow](#8-training-mlflow-and-model-versioning)
and [Deployment](#13-deployment).

---

## 7. Dashboard user guide

Launch with `streamlit run dashboard/app.py`. Pages appear in the **left
sidebar** (hamburger menu on phones) under two groups:

### Operations

1. **Single Prediction** — enter the 5 sensors (each label shows the SI unit
   and allowed range) and the machine type, set the risk threshold, and click
   *Run prediction*. You get the decision, probability, anomaly score,
   agreement, detector flag, and an expandable SHAP table.

2. **Real-Time Manual Monitoring** — the same form inside a `st.form`; every
   submitted reading is validated, predicted (with SHAP), and appended to a
   session log with a timestamp and torque-drift σ vs the training
   reference. Below the form: reading counts, latest state, and two trend
   charts (failure probability, torque). Inputs pre-fill from your last
   reading so repeated checks are fast.

3. **Batch CSV Prediction** — upload an AI4I-format CSV. All rows are
   validated (per-issue warnings shown), scored in one pass, and displayed
   with decision columns; a results summary, decision-distribution bar
   chart, and a **Download results CSV** button are provided. Results are
   cached in the session for the two analytics pages.

### Intelligence

4. **Machine Health Analytics** — fleet view over the last batch: counts,
   decision share pie, risk-by-type histogram, failure-probability
   distribution, and per-feature histograms colored by decision.

5. **Explainability (SHAP)** — global model feature importance (bar chart)
   and a guide to reading SHAP values. Per-machine SHAP appears on pages 1–2.

6. **System Monitoring** — per-feature input drift (σ units) of the last
   batch vs the training reference (stored at train time in
   `models/latest/reference_stats.joblib`), with a DRIFT/OK status chart,
   plus the classifier ↔ detector disagreement rate.

---

## 8. Training, MLflow and model versioning

`train_cli.py` wraps `train()` from `src/.../train.py`:

```powershell
python train_cli.py --data <csv> --model {random_forest|xgboost}
```

Each run:

1. Loads + validates the CSV, engineers features, splits 80/20 stratified.
2. Trains the classifier and the IsolationForest (healthy rows only).
3. Logs params, metrics, and both models to **MLflow**
   (tracking URI `sqlite:///mlflow.db`, experiment
   `predict_industrial_quality_ai`).
4. Saves timestamped artifacts:
   `models/classifier_<type>_<timestamp>.joblib`,
   `models/anomaly_detector_<timestamp>.joblib`.
5. Refreshes `models/latest/` (`classifier.joblib`, `anomaly_detector.joblib`,
   `registry.joblib`, `reference_stats.joblib`) — this is what the dashboard loads.

Browse experiments:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

> Note: `models/` is git-ignored. For Streamlit Cloud, either commit a copy
> of `models/latest/` or add a bootstrap train step (see Deployment).

---

## 9. Logging

`src/.../logging_config.py` provides `get_logger(name)` used by every module.
Logs are written to **both** `logs/app.log` and stdout with the format:

```
2026-10-19 01:14:35 | INFO | predict_industrial_quality_ai.decision | Predicted 1 record(s); ...
```

Logged events include validation results, training progress, final test
metrics, and per-call prediction summaries.

---

## 10. Testing

```powershell
python -m pytest tests -q
```

**14 tests**, all passing:

| File | Covers |
|---|---|
| `tests/test_validation.py` | valid frames pass; missing columns rejected; out-of-range dropped; missing values dropped; duplicates removed; invalid type dropped; all-invalid raises; single-record checks |
| `tests/test_pipeline_and_decision.py` | domain feature derivation; pipeline fit/predict; preprocessor handles unseen categories; decision states & priorities & SHAP payload on a risky vs healthy machine; full-batch decision shape |

The dashboard is additionally verified headlessly with Streamlit's official
`AppTest` framework (page rendering + prediction/submission flows).

---

## 11. Project structure

```
Predict_Industrail_Quality_AI/
├── .streamlit/
│   └── config.toml                  # dark theme tokens (base, colors)
├── dashboard/
│   ├── app.py                       # entry: st.navigation multipage
│   ├── ui_common.py                 # cached models, sensor grid, decision card
│   └── pages/
│       ├── single_prediction.py
│       ├── live_monitoring.py
│       ├── batch_csv.py
│       ├── health_analytics.py
│       ├── explainability.py
│       └── system_monitoring.py
├── data/
│   └── ai4i_synthetic.csv           # generated stand-in dataset
├── src/predict_industrial_quality_ai/
│   ├── __init__.py
│   ├── validation.py                # Layer 2
│   ├── features.py                  # Layer 3
│   ├── train.py                     # Layer 4
│   ├── decision.py                  # Layer 5 (incl. SHAP)
│   └── logging_config.py            # Layer 7
├── scripts/
│   └── generate_data.py             # synthetic AI4I generator
├── tests/
│   ├── conftest.py                  # puts src/ on sys.path
│   ├── test_validation.py
│   └── test_pipeline_and_decision.py
├── train_cli.py                     # training entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

Runtime-generated (git-ignored): `models/`, `mlruns/`, `mlflow.db`,
`logs/`, `__pycache__/`.

---

## 12. Configuration reference

| What | Where | How to change |
|---|---|---|
| Allowed input ranges | `validation.py` → `RANGES` | edit the dict; the dashboard labels update automatically |
| Grey band bounds | `decision.py` → `GREY_BAND_LOW/HIGH` | defaults 0.4 / 0.7 |
| Classifier hyperparameters | `train.py` → `make_classifier()` | RF kwargs or XGB kwargs |
| Anomaly contamination | `train.py` → `IsolationForest(...)` | default 0.05 |
| Theme / colors | `.streamlit/config.toml` | `primaryColor`, backgrounds, text |
| MLflow backend | `train.py` → `mlflow.set_tracking_uri()` | default `sqlite:///mlflow.db` |
| Default risk threshold | dashboard sliders | per-page, 0.50–0.95 |

---

## 13. Deployment

### Local

```powershell
streamlit run dashboard/app.py
```

### Streamlit Community Cloud (zero manual config)

The app is **self-bootstrapping**: on first boot, `ensure_models()` in
`dashboard/ui_common.py` checks for `models/latest/` artifacts; if they are
missing (they are git-ignored on the cloud instance) it generates the data
if needed and trains the Random Forest + Isolation Forest automatically
(~45 s behind a spinner). No build commands, no committed model files.

**Deploy steps:**

1. Push this repository to GitHub (all source, `data/`, `requirements.txt`,
   `.streamlit/config.toml` are tracked):
   ```powershell
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** →
   pick the repo/branch:
   - **Main file path:** `dashboard/app.py`
   - **Python version:** 3.11+ (defaults are fine)
   - **Requirements file:** `requirements.txt` (auto-detected)
3. Deploy. First boot shows *"Preparing models (first run trains
   automatically)…"* while training runs; after that, models load in <1 s.
4. Optional advanced settings → **Secrets** are not required (the app has no
   secret inputs); MLflow tracking stays local to the instance.

**What to know for the cloud:**

| Concern | Behavior on Streamlit Cloud |
|---|---|
| `models/` git-ignored | auto-recreated by `ensure_models()` on first boot |
| `mlflow.db` / `mlruns/` | ephemeral per instance — experiment history resets on redeploy; host MLflow externally for persistence |
| `logs/app.log` | written to instance disk, ephemeral |
| Instance restarts | model bootstrap re-runs (also ~45 s, once per fresh instance) |
| Warm app loading | lazy model load via `st.cache_resource` — first prediction instant thereafter |
| Ephemeral disk | git-ignored artifacts (`models/`, logs, MLflow) are never lost from git because they are regenerable |

### Running from any working directory

All dashboard pages resolve paths relative to the repository root via
`pathlib`, and `ui_common.py` puts `src/` on `sys.path` — the app starts
correctly regardless of the launch cwd.

---

## 14. Troubleshooting FAQ

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: predict_industrial_quality_ai` | run `streamlit run dashboard/app.py` from the repo root (pages self-configure the path, but models are resolved relative to root); or set `PYTHONPATH=src` |
| Values invisible in input fields | do a hard refresh (`Ctrl+F5`) so stale cached CSS from older versions is purged; the current UI uses only Streamlit-native widgets |
| `Model not available` / models missing | the dashboard auto-trains on first boot; if you run headless code, train first: `python scripts/generate_data.py && python train_cli.py --data data/ai4i_synthetic.csv` |
| All rows rejected by validation | check your CSV column names match the AI4I schema exactly (e.g. `Air temperature [K]` with the bracket unit); see the ranges table |
| SHAP import error (numba) | `pip install numba shap` |
| MLflow "file store maintenance mode" | the project already uses the SQLite backend; re-run training |
| Dashboard slow on first load | the first prediction triggers lazy model loading + SHAP initialization; subsequent interactions are fast |

---

## 15. Roadmap

- [ ] Retraining CLI flag for the real UCI dataset with drift-aware thresholds
- [ ] Model registry promotion (MLflow Model Registry stages: Staging → Production)
- [ ] Prediction audit log persisted to SQLite for long-term monitoring
- [ ] Optional login for multi-user deployments
- [ ] Docker image for one-command deployment

---

## License & attribution

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository,
CC BY 4.0). This code is provided as-is for educational and prototyping use.
