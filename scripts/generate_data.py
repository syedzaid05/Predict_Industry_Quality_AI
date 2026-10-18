"""Generates a synthetic AI4I-2020-style CSV when the real dataset is unavailable."""
import argparse
import numpy as np
import pandas as pd


def generate(n=5000, seed=42, failure_frac=0.033, path="data/ai4i_synthetic.csv"):
    rng = np.random.default_rng(seed)
    type_ = rng.choice(["L", "M", "H"], n, p=[0.6, 0.3, 0.1])
    air = rng.normal(298.15 + np.where(type_ == "H", 1.5, 0.3), 2.0, n)
    proc = air + rng.normal(10.5, 1.2, n)
    rpm = rng.normal(1500 + np.where(type_ == "H", -75, 0), 90, n)
    torque = rng.normal(40 + np.where(rpm < 1350, 12, 0), 8, n)
    wear = rng.integers(0, 280, n).astype(float)

    twf = torque > 62
    hdf = (proc - air > 11.8) & (rpm < 1350)
    pwf = ((torque * rpm) > 90000) | (torque < 14)
    osf = wear > 252
    twf = twf | ((wear > 240) & (torque > 55))
    fail = (twf | hdf | pwf | osf).astype(int)
    fail = np.maximum(fail, (rng.random(n) < failure_frac).astype(int))
    # noise: some flagged-but-healthy, some silent failures
    fail = np.where(rng.random(n) < 0.01, 1 - fail, fail)

    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{i}" for t, i in zip(type_, np.arange(1, n + 1))],
        "Type": type_,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm.round(0),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear,
        "Machine failure": fail,
        "TWF": 0, "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": 0,
    })
    df.to_csv(path, index=False)
    print(f"Wrote {path}: {n} rows, failure rate {fail.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--path", default="data/ai4i_synthetic.csv")
    args = ap.parse_args()
    generate(args.n, path=args.path)
