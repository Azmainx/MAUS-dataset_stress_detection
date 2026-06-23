"""
evaluate.py — metrics, confusion matrix, ROC for the trained model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
)

import config as cfg


def evaluate(model, X_test, y_test, task="binary"):
    if task == "binary":
        probs = model.predict(X_test).ravel()
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
    else:
        probs = model.predict(X_test)
        preds = np.argmax(probs, axis=1)
        auc = None

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="binary" if task == "binary" else "macro")

    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}" + (f"  AUC: {auc:.4f}" if auc else ""))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    fig.savefig(os.path.join(cfg.PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if task == "binary":
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        fig.savefig(os.path.join(cfg.PLOTS_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {"accuracy": acc, "f1": f1, "auc": auc}


if __name__ == "__main__":
    import tensorflow as tf
    import model  # noqa: F401 - ensures AttentionLayer gets registered before loading
    model_obj = tf.keras.models.load_model(
        os.path.join(cfg.MODEL_DIR, "best_model.keras"),
    )
    data = np.load(os.path.join(cfg.OUTPUT_DIR, "test_set.npz"))
    evaluate(model_obj, data["X"], data["y"], task="binary")
