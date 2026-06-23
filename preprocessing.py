"""
preprocessing.py — filtering, normalization, windowing & labeling for one MAUS subject.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config as cfg


def _butter_filter(x, low=None, high=None, fs=cfg.FS, order=cfg.FILTER_ORDER, btype="band"):
    nyq = fs / 2.0
    if btype == "band":
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
    elif btype == "low":
        b, a = butter(order, high / nyq, btype="low")
    else:
        raise ValueError(btype)
    return filtfilt(b, a, x)


def filter_channel(x, kind):
    """kind in {'ecg','ppg','gsr'}"""
    x = np.nan_to_num(x, nan=np.nanmean(x))
    if kind == "ecg":
        lo, hi = cfg.ECG_BANDPASS
        return _butter_filter(x, lo, hi, btype="band")
    elif kind == "ppg":
        lo, hi = cfg.PPG_BANDPASS
        return _butter_filter(x, lo, hi, btype="band")
    elif kind == "gsr":
        return _butter_filter(x, high=cfg.GSR_LOWPASS, btype="low")
    raise ValueError(kind)


def zscore(x, mu=None, sigma=None):
    mu = np.mean(x) if mu is None else mu
    sigma = np.std(x) + 1e-8 if sigma is None else sigma
    return (x - mu) / sigma, mu, sigma


def load_subject_raw(subject_dir):
    """Returns dict {trial_col: {'ecg':arr, 'ppg':arr, 'gsr':arr}}"""
    ecg = pd.read_csv(os.path.join(subject_dir, "inf_ecg.csv"))
    ppg = pd.read_csv(os.path.join(subject_dir, "inf_ppg.csv"))
    gsr = pd.read_csv(os.path.join(subject_dir, "inf_gsr.csv"))

    out = {}
    for col in cfg.TRIAL_COLUMNS:
        out[col] = {
            "ecg": ecg[col].values.astype(float),
            "ppg": ppg[col].values.astype(float),
            "gsr": gsr[col].values.astype(float),
        }
    return out


def preprocess_subject(subject_dir, task="binary"):
    """
    Full pipeline for one subject:
      load -> filter -> per-subject z-score -> window -> label
    Returns X (n_windows, window_len, 3), y (n_windows,)
    """
    raw = load_subject_raw(subject_dir)
    label_map = cfg.TRIAL_LABEL_BINARY if task == "binary" else cfg.TRIAL_LABEL_MULTICLASS

    # --- filter each trial/channel ---
    filtered = {}
    for col, sig in raw.items():
        filtered[col] = {
            "ecg": filter_channel(sig["ecg"], "ecg"),
            "ppg": filter_channel(sig["ppg"], "ppg"),
            "gsr": filter_channel(sig["gsr"], "gsr"),
        }

    # --- subject-level normalization stats (computed across all trials) ---
    all_ecg = np.concatenate([filtered[c]["ecg"] for c in cfg.TRIAL_COLUMNS])
    all_ppg = np.concatenate([filtered[c]["ppg"] for c in cfg.TRIAL_COLUMNS])
    all_gsr = np.concatenate([filtered[c]["gsr"] for c in cfg.TRIAL_COLUMNS])
    _, mu_e, sd_e = zscore(all_ecg)
    _, mu_p, sd_p = zscore(all_ppg)
    _, mu_g, sd_g = zscore(all_gsr)

    X_list, y_list = [], []
    win, stride = cfg.WINDOW_LEN, cfg.STRIDE_LEN

    for col in cfg.TRIAL_COLUMNS:
        ecg_n, _, _ = zscore(filtered[col]["ecg"], mu_e, sd_e)
        ppg_n, _, _ = zscore(filtered[col]["ppg"], mu_p, sd_p)
        gsr_n, _, _ = zscore(filtered[col]["gsr"], mu_g, sd_g)

        n = len(ecg_n)
        label = label_map[col]

        for start in range(0, n - win + 1, stride):
            seg = np.stack(
                [ecg_n[start:start + win], ppg_n[start:start + win], gsr_n[start:start + win]],
                axis=-1,
            )  # (win, 3)
            if cfg.DOWNSAMPLE_FACTOR > 1:
                seg = seg[::cfg.DOWNSAMPLE_FACTOR]
            X_list.append(seg)
            y_list.append(label)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    return X, y


if __name__ == "__main__":
    # quick smoke test on the sample subject placed directly at DATA_DIR root
    X, y = preprocess_subject(cfg.DATA_DIR, task="binary")
    print("X shape:", X.shape, "y shape:", y.shape, "label balance:", np.bincount(y))
