"""
dataset_builder.py — loop over all subject folders, preprocess each, and build
subject-wise train/val/test tensors (no window-level leakage across splits).
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

import config as cfg
from preprocessing import preprocess_subject


def list_subject_dirs(data_dir):
    """
    Discover subject folders by checking for required CSV files.
    
    Lists every entry directly under data_dir and keeps only:
      - directories that contain inf_ecg.csv, inf_ppg.csv, and inf_gsr.csv
    
    Returns a sorted list of full subject directory paths.
    Also returns subject IDs (folder names) alongside paths for debugging/joins.
    """
    required_files = ["inf_ecg.csv", "inf_ppg.csv", "inf_gsr.csv"]
    subs = []
    
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Data directory does not exist: {data_dir}")
    
    for d in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, d)
        if not os.path.isdir(full_path):
            continue
        
        # Check if all required files are present
        has_all_files = all(os.path.exists(os.path.join(full_path, f)) for f in required_files)
        if has_all_files:
            subs.append(full_path)
    
    return subs


def build_dataset(data_dir=cfg.DATA_DIR, task="binary"):
    subject_dirs = list_subject_dirs(data_dir)
    if len(subject_dirs) == 0:
        raise RuntimeError(f"No subject folders with inf_ecg.csv found under {data_dir}")

    per_subject_X, per_subject_y = [], []
    for sd in subject_dirs:
        X, y = preprocess_subject(sd, task=task)
        per_subject_X.append(X)
        per_subject_y.append(y)

    if len(subject_dirs) == 1:
        # Only 1 valid subject found -> log a smoke-test warning
        print("[WARN] Only 1 subject found — using subject-wise split with 1 subject (smoke test).")
        # For a true single-subject evaluation (window-level split), use:
        #   X, y = per_subject_X[0], per_subject_y[0]
        #   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, ...)
        # But still use subject-wise split for consistency

    # --- proper subject-wise split ---
    idx = np.arange(len(subject_dirs))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=cfg.RANDOM_SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=cfg.RANDOM_SEED)

    def gather(indices):
        Xs = np.concatenate([per_subject_X[i] for i in indices])
        ys = np.concatenate([per_subject_y[i] for i in indices])
        return Xs, ys

    return gather(train_idx), gather(val_idx), gather(test_idx)


if __name__ == "__main__":
    (Xtr, ytr), (Xva, yva), (Xte, yte) = build_dataset()
    print("Train:", Xtr.shape, np.bincount(ytr))
    print("Val:  ", Xva.shape, np.bincount(yva))
    print("Test: ", Xte.shape, np.bincount(yte))
