"""
shap_analysis.py — explain the CNN-BiLSTM-Attention model with SHAP GradientExplainer.

Produces:
  1. per-channel (ECG/PPG/GSR) importance bar chart
  2. per-time-step SHAP heatmap overlaid on a sample window
  3. a SHAP summary plot across many test windows
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf

import config as cfg

CHANNEL_NAMES = ["ECG", "PPG", "GSR"]


def run_shap(model, X_background, X_explain, n_background=100, n_explain=50):
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    bg_idx = rng.choice(len(X_background), size=min(n_background, len(X_background)), replace=False)
    ex_idx = rng.choice(len(X_explain), size=min(n_explain, len(X_explain)), replace=False)

    background = X_background[bg_idx]
    samples = X_explain[ex_idx]

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(samples)
    # shap_values: list (n_outputs) of arrays (n_samples, T, C) for multiclass,
    # or a single array (n_samples, T, C, 1) for binary sigmoid output.
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values
    sv = np.squeeze(np.array(sv))  # (n_samples, T, C)
    return sv, samples


def plot_channel_importance(sv, save_path):
    # mean(|shap|) over samples and time, per channel
    importance = np.mean(np.abs(sv), axis=(0, 1))  # (C,)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(CHANNEL_NAMES[: len(importance)], importance, color=["#d62728", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Channel importance (ECG vs PPG vs GSR)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return importance


def plot_time_heatmap(sv, samples, save_path, sample_idx=0):
    # sv, samples: (n_samples, T, C)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    t = np.arange(sv.shape[1])
    for c, name in enumerate(CHANNEL_NAMES[: sv.shape[2]]):
        ax = axes[c]
        ax.plot(t, samples[sample_idx, :, c], color="black", lw=1, label="signal (z-scored)")
        sc = ax.scatter(
            t, samples[sample_idx, :, c],
            c=sv[sample_idx, :, c], cmap="coolwarm", s=8,
            vmin=-np.max(np.abs(sv[sample_idx])), vmax=np.max(np.abs(sv[sample_idx])),
        )
        ax.set_ylabel(name)
    axes[-1].set_xlabel("Time step (within window)")
    fig.colorbar(sc, ax=axes, label="SHAP value", fraction=0.02)
    fig.suptitle("SHAP attribution over time, per channel (example window)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(sv, save_path):
    # flatten time+channel into "features" for a standard SHAP summary bar
    n, T, C = sv.shape
    flat = np.abs(sv).reshape(n, T * C)
    mean_abs = flat.mean(axis=0).reshape(T, C).mean(axis=0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(CHANNEL_NAMES[: C], mean_abs)
    ax.set_xlabel("Mean |SHAP value| (aggregated)")
    ax.set_title("Overall modality contribution")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    import model as model_module  # noqa: F401 - ensures AttentionLayer gets registered
    model = tf.keras.models.load_model(os.path.join(cfg.MODEL_DIR, "best_model.keras"))
    data = np.load(os.path.join(cfg.OUTPUT_DIR, "test_set.npz"))
    X_test, y_test = data["X"], data["y"]

    sv, samples = run_shap(model, X_background=X_test, X_explain=X_test)

    plot_channel_importance(sv, os.path.join(cfg.PLOTS_DIR, "shap_channel_importance.png"))
    plot_time_heatmap(sv, samples, os.path.join(cfg.PLOTS_DIR, "shap_time_heatmap.png"))
    plot_summary(sv, os.path.join(cfg.PLOTS_DIR, "shap_summary.png"))
    print("SHAP plots saved to", cfg.PLOTS_DIR)


if __name__ == "__main__":
    main()
