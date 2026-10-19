"""Layer 2 - Data validation.

Validates AI4I 2020 schema: types, missing values, range checks,
duplicates and overall rejection of invalid input blocks.
"""
from __future__ import annotations

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLS = ["Type"]
REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS

RANGES = {
    "Air temperature [K]": (250.0, 400.0),
    "Process temperature [K]": (250.0, 420.0),
    "Rotational speed [rpm]": (500.0, 3500.0),
    "Torque [Nm]": (0.0, 120.0),
    "Tool wear [min]": (0.0, 350.0),
}
VALID_TYPES = {"L", "M", "H"}


class ValidationError(Exception):
    """Raised when input cannot be used for prediction."""


# ---------------------------------------------------------- universal CSV intake

_SYNONYMS = {
    "Air temperature [K]": (
        "airtemp", "airtemperature", "airtemperaturek", "airtempk", "air", "tempair",
    ),
    "Process temperature [K]": (
        "processtemp", "processtemperature", "processtemperaturek", "processtempk",
        "processtemperture", "ptemp",
    ),
    "Rotational speed [rpm]": (
        "rotationspeed", "rotationalspeed", "rotationalrpm", "rpm", "speed", "speedrpm",
    ),
    "Torque [Nm]": ("torque", "torquenm", "tq", "torquenmvalue"),
    "Tool wear [min]": ("toolwear", "toolwearmin", "toolwearminvalue", "wear", "toolwearmins"),
    "Type": ("type", "machinetype", "qualityvariant", "variant", "quality", "qualitytype"),
}
_TYPE_VALUES = {
    "L": "L", "LOW": "L", "LOWQ": "L",
    "M": "M", "MED": "M", "MEDIUM": "M", "MEDQ": "M",
    "H": "H", "HIGH": "H", "HIGHQ": "H",
}


def _norm_header(h: str) -> str:
    """'Air temperature [K]' -> 'airtemperaturekish' style lowercase key."""
    s = str(h).strip().lower()
    for junk in ("(", ")", "[", "]", "{", "}", "°", '"', "'"):
        s = s.replace(junk, " ")
    s = "".join(s.split())
    s = s.replace("celsius", "").replace("kelvin", "").replace("k-", "").replace("k_", "")
    return s.strip("-_.k")


def standardize_columns(df: pd.DataFrame, context: str = "input") -> tuple[pd.DataFrame, list[str]]:
    """Map a CSV with (almost) any header names onto the AI4I schema.

    Returns (standardized_df, notes). Raises ValidationError only when a
    required sensor column genuinely cannot be found.
    """
    notes: list[str] = []
    norm = {c: _norm_header(c) for c in df.columns}
    out: dict[str, pd.Series] = {}

    for target, keys in _SYNONYMS.items():
        if target in df.columns:  # already canonical
            out[target] = df[target]
            continue
        match = next((c for c, k in norm.items() if k in keys), None)
        if match is not None:
            out[target] = df[match]
            if match != target:
                notes.append(f"Mapped column '{match}' → '{target}'")
        elif target == "Type":
            out[target] = "M"  # sensible default for legacy files without Type
            notes.append("No type column found — defaulted every row to 'M' (medium variant)")

    missing = [c for c in NUMERIC_COLS if c not in out]
    if missing:
        raise ValidationError(
            f"Required sensor column(s) not found: {missing}. "
            f"Columns seen: {list(df.columns)[:12]}"
        )
    if "Type" not in out:
        out["Type"] = "M"
        notes.append("Type column missing — defaulted to 'M'")

    std = pd.DataFrame(out)
    std.index = df.index

    # flexible Type values (low/medium/high, casing, trailing spaces)
    std["Type"] = std["Type"].astype(str).str.strip().str.upper()
    std["Type"] = std["Type"].map(lambda v: _TYPE_VALUES.get(v, v))
    unmapped = ~std["Type"].isin(VALID_TYPES)
    if unmapped.any():
        notes.append(f"{int(unmapped.sum())} row(s) with unrecognized Type — set to 'M'")
        std.loc[unmapped, "Type"] = "M"

    logger.info("Standardized %s: %d input column(s) -> schema", context, len(df.columns))
    return std, notes


def validate_dataframe(df: pd.DataFrame, context: str = "input") -> tuple[pd.DataFrame, list[str]]:
    """Validate a DataFrame. Returns (clean_df, issues).

    Rows that are structurally unfixable (missing/non-numeric/out-of-range)
    are dropped; remaining issues are reported as warnings.
    Raises ValidationError if no valid rows remain or schema is wrong.
    """
    issues: list[str] = []

    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame")

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")

    df = df.loc[:, REQUIRED_COLS].copy()

    # Duplicates
    dup_mask = df.duplicated()
    if dup_mask.any():
        issues.append(f"Removed {int(dup_mask.sum())} duplicate row(s)")
        df = df[~dup_mask]

    # Non-numeric / coercible
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_numeric = df[NUMERIC_COLS].isna().sum()
    for col, n in bad_numeric.items():
        if n:
            issues.append(f"{n} row(s) with invalid/missing value in '{col}'")

    # Categorical cleanup
    df["Type"] = df["Type"].astype(str).str.strip().str.upper()
    invalid_type = ~df["Type"].isin(VALID_TYPES)
    if invalid_type.any():
        issues.append(f"{int(invalid_type.sum())} row(s) with invalid 'Type' (expected L/M/H)")

    # Range checks
    range_mask = pd.Series(False, index=df.index)
    for col, (lo, hi) in RANGES.items():
        m = ~df[col].between(lo, hi)
        if m.any():
            issues.append(f"{int(m.sum())} row(s) out of range for '{col}' (allowed {lo}-{hi})")
        range_mask |= m

    missing_mask = df[NUMERIC_COLS].isna().any(axis=1)
    drop_mask = invalid_type | missing_mask | range_mask
    if drop_mask.any():
        n = int(drop_mask.sum())
        issues.append(f"Dropped {n} of {len(df)} row(s) as invalid")
        df = df[~drop_mask]

    if df.empty:
        raise ValidationError("All rows were rejected by validation — no usable input")

    logger.info("Validated %s: %d row(s) accepted, %d issue(s)", context, len(df), len(issues))
    return df, issues


def validate_single_record(record: dict) -> dict:
    """Validate one manual-input record; raises ValidationError on failure."""
    missing = [c for c in REQUIRED_COLS if c not in record]
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    clean: dict = {"Type": str(record["Type"]).strip().upper()}
    if clean["Type"] not in VALID_TYPES:
        raise ValidationError(f"Invalid Type '{clean['Type']}' (expected L/M/H)")
    for col in NUMERIC_COLS:
        try:
            val = float(record[col])
        except (TypeError, ValueError):
            raise ValidationError(f"'{col}' must be numeric")
        lo, hi = RANGES[col]
        if not (lo <= val <= hi):
            raise ValidationError(f"'{col}'={val} outside allowed range {lo}-{hi}")
        clean[col] = val
    return clean
