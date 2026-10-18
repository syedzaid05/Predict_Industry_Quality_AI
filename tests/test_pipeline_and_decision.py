import numpy as np
import pandas as pd

from predict_industrial_quality_ai.features import (
    build_preprocessor,
    build_pipeline,
    add_engineered_features,
)
from predict_industrial_quality_ai.train import make_classifier
from predict_industrial_quality_ai.decision import Predictor
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def make_train_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Air temperature [K]": rng.normal(298, 2, n),
            "Process temperature [K]": rng.normal(308, 2, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": rng.normal(45, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n).astype(float),
            "Type": rng.choice(["L", "M", "H"], n),
        }
    )
    df["Machine failure"] = (
        (df["Torque [Nm]"] > 65) | (df["Tool wear [min]"] > 200) | (df["Rotational speed [rpm]"] < 1300)
    ).astype(int)
    return df


def test_engineered_features_added():
    out = add_engineered_features(make_train_df(10))
    for col in ["Temp diff [K]", "Power [W]", "Wear x Torque"]:
        assert col in out.columns
    assert (out["Power [W]"] > 0).mean() > 0.9


def test_pipeline_fit_predict():
    X = make_train_df(400)
    y = X.pop("Machine failure")
    pipe = build_pipeline(make_classifier("random_forest"))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X.head(5))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_preprocessor_is_train_fitted():
    X = make_train_df(50)
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape[0] == 50
    # unseen category does not crash thanks to handle_unknown="ignore"
    X2 = X.copy()
    X2["Type"] = "Z"
    assert pre.transform(X2).shape[0] == 50


def test_decision_layer_states_and_explain():
    df = make_train_df(400)
    y = df.pop("Machine failure")
    pipe = build_pipeline(make_classifier("random_forest"))
    pipe.fit(df, y)
    iso = IsolationForest(random_state=42, contamination=0.1).fit(
        pipe.named_steps["preprocessor"].transform(df[y == 0])
    )
    pred = Predictor(pipe, iso)

    healthy = pd.DataFrame(
        [{"Air temperature [K]": 298, "Process temperature [K]": 308, "Rotational speed [rpm]": 1550,
          "Torque [Nm]": 30, "Tool wear [min]": 5, "Type": "L"}]
    )
    risky = pd.DataFrame(
        [{"Air temperature [K]": 295, "Process temperature [K]": 310, "Rotational speed [rpm]": 1200,
          "Torque [Nm]": 90, "Tool wear [min]": 240, "Type": "L"}]
    )
    out = pred.predict(pd.concat([healthy, risky]), explain=True)
    assert len(out) == 2
    assert out[0]["failure_probability"] < out[1]["failure_probability"]
    assert out[1]["decision"] in {"FAILURE_RISK", "WARNING", "INSPECT"}
    assert out[0]["maintenance_priority"].startswith("P4") or out[0]["maintenance_priority"].startswith("P3")
    assert len(out[0]["shap"]) >= 2  # SHAP drivers present


def test_batch_decisions_shape():
    df = make_train_df(120)
    y = df.pop("Machine failure")
    pipe = build_pipeline(make_classifier("random_forest"))
    pipe.fit(df, y)
    iso = IsolationForest(random_state=42).fit(pipe.named_steps["preprocessor"].transform(df[y == 0]))
    pred = Predictor(pipe, iso)
    out = pred.predict(df, explain=False)
    assert len(out) == 120
    assert all(set(d) >= {"decision", "failure_probability", "anomaly_score"} for d in out)
