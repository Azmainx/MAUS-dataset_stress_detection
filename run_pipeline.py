"""
run_pipeline.py — single entrypoint: preprocess -> train -> evaluate -> SHAP.

Usage:
    python run_pipeline.py --data_dir ./data --task binary --epochs 50
"""

import argparse
import config as cfg
from train import train
from evaluate import evaluate
import shap_analysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=cfg.RAW_DATA_DIR)
    p.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    p.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    args = p.parse_args()

    print("=== Step 1-4: preprocessing + training ===")
    model, history, (Xte, yte) = train(args.data_dir, args.task, args.epochs)

    print("=== Step 5: evaluation ===")
    evaluate(model, Xte, yte, task=args.task)

    print("=== Step 6: SHAP analysis ===")
    shap_analysis.main()

    print("Done. See", cfg.PLOTS_DIR, "and", cfg.MODEL_DIR)


if __name__ == "__main__":
    main()
