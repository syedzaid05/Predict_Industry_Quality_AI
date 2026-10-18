# Predict Industrial Quality AI

End-to-end predictive-maintenance ML system built on the **AI4I 2020** schema.

```
Data sources (CSV / manual / upload)
  → Validation (schema, missing, ranges, duplicates, rejection)
    → Feature engineering (encodings + domain transforms, train-fitted pipeline)
      → ML engine: Failure classifier (RandomForest / XGBoost) + Anomaly detector (Isolation Forest)
        → Decision layer (risk threshold · grey band · model disagreement · maintenance priority)
          → SHAP explainability
            → Streamlit dashboard (single · batch · analytics · explainability · monitoring)
```

## Quickstart

```powershell
pip install -r requirements.txt

# 1. Data (real AI4I 2020 CSV, or a synthetic stand-in)
python scripts/generate_data.py            # → data/ai4i_synthetic.csv

# 2. Train (raises RandomForest or XGBoost classifier + IsolationForest, logs to MLflow)
python train_cli.py --data data/ai4i_synthetic.csv --model random_forest   # or xgboost

# 3. Run tests
python -m pytest tests -q

# 4. Dashboard
streamlit run dashboard/app.py
```

## Project layout

```
src/predict_industrial_quality_ai/
  validation.py      Layer 2 - schema/missing/range/duplicate checks, input rejection
  features.py        Layer 3 - FeatureEngineer transformer + fitted preprocessor (scaler, OHE)
  train.py           Layer 4 - RF/XGB training, IsolationForest, MLflow logging, model versioning
  decision.py        Layer 5 - risk thresholds, grey band, disagreement logic, priorities, SHAP
  logging_config.py  Layer 7 - rotating file+stdout logging
dashboard/app.py     Layer 6 - 5-page Streamlit app
tests/               pytest: validation, pipeline, decision-layer, SHAP
scripts/generate_data.py  synthetic AI4I-format generator
models/latest/       versioned artifacts (classifier, anomaly detector, registry, reference stats)
mlflow.db + mlruns/  MLflow tracking (sqlite backend)
```

## Dashboard pages
- **Single Prediction** — manual sensor input, risk-threshold slider, SHAP drivers, priority
- **Batch CSV Prediction** — upload AI4I-format CSV, download annotated results
- **Machine Health Analytics** — decision share, risk by type, probability & feature histograms
- **Explainability** — model feature importance + how to read SHAP values
- **Monitoring** — input drift vs training reference stats, classifier ↔ detector disagreement rate

## MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
