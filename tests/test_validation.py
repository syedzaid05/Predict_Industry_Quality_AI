import numpy as np
import pandas as pd
import pytest

from predict_industrial_quality_ai.validation import ValidationError, validate_dataframe, validate_single_record


def base_row(**over):
    row = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
        "Type": "L",
    }
    row.update(over)
    return row


def test_valid_frame_passes():
    df = pd.DataFrame([
        base_row(),
        base_row(**{"Torque [Nm]": 20.0, "Tool wear [min]": 3.0, "Type": "H"}),
        base_row(**{"Rotational speed [rpm]": 1400.0, "Tool wear [min]": 50.0}),
    ])
    clean, issues = validate_dataframe(df)
    assert len(clean) == 3
    assert issues == []


def test_missing_column_rejected():
    df = pd.DataFrame([{k: v for k, v in base_row().items() if k != "Torque [Nm]"}])
    with pytest.raises(ValidationError):
        validate_dataframe(df)


def test_out_of_range_dropped():
    df = pd.DataFrame([base_row(**{"Torque [Nm]": 999.0}), base_row()])
    clean, issues = validate_dataframe(df)
    assert len(clean) == 1
    assert any("range" in i for i in issues)


def test_missing_values_dropped():
    df = pd.DataFrame([base_row(**{"Tool wear [min]": None}), base_row()])
    clean, issues = validate_dataframe(df)
    assert len(clean) == 1


def test_duplicates_removed():
    df = pd.DataFrame([base_row(), base_row()])
    clean, issues = validate_dataframe(df)
    assert len(clean) == 1
    assert any("duplicate" in i.lower() for i in issues)


def test_invalid_type_dropped():
    df = pd.DataFrame([base_row(Type="X"), base_row()])
    clean, _ = validate_dataframe(df)
    assert len(clean) == 1


def test_all_invalid_raises():
    with pytest.raises(ValidationError):
        validate_dataframe(pd.DataFrame([base_row(**{"Torque [Nm]": 500.0})]))


def test_single_record_valid():
    rec = validate_single_record(base_row())
    assert rec["Type"] == "L"
    assert np.isfinite(rec["Torque [Nm]"])


def test_single_record_bad_range_raises():
    with pytest.raises(ValidationError):
        validate_single_record(base_row(**{"Rotational speed [rpm]": 9000.0}))
