"""CLI training script."""
import argparse
import sys

sys.path.insert(0, "src")

from predict_industrial_quality_ai.train import train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to AI4I 2020 CSV")
    ap.add_argument("--model", default="random_forest", choices=["random_forest", "xgboost"])
    args = ap.parse_args()
    m = train(args.data, args.model)
    print("Training complete:")
    for k, v in m.items():
        print(f"  {k}: {v}")
