"""
config.py — all hyperparameters and paths for the MAUS CNN-BiLSTM-Attention pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAUS_ROOT = r"E:\Thesis\MAUS"
RAW_DATA_DIR = os.path.join(MAUS_ROOT, "Data", "Raw_data")
IBI_SEQUENCE_DIR = os.path.join(MAUS_ROOT, "Data", "IBI_sequence")        # not used yet
SUBJECTIVE_RATING_DIR = os.path.join(MAUS_ROOT, "Subjective_rating")      # not used yet

# Backward-compatible alias: CLI args still use --data_dir (defaults to RAW_DATA_DIR)
DATA_DIR = RAW_DATA_DIR

OUTPUT_DIR = "./outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

for d in (OUTPUT_DIR, MODEL_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Signal / preprocessing
# ---------------------------------------------------------------------------
FS = 256                     # sampling rate (Hz) of the Procomp Infiniti ECG/PPG/GSR
WINDOW_SEC = 10               # window length in seconds
WINDOW_STRIDE_SEC = 5          # stride (50% overlap)
WINDOW_LEN = WINDOW_SEC * FS    # samples per window (2560)
STRIDE_LEN = WINDOW_STRIDE_SEC * FS

DOWNSAMPLE_FACTOR = 5          # 2560 -> 512 samples/window fed to the CNN (set 1 to disable)

ECG_BANDPASS = (0.5, 40.0)
PPG_BANDPASS = (0.5, 8.0)
GSR_LOWPASS = 1.0
FILTER_ORDER = 4

TRIAL_COLUMNS = [
    "Trial 1:0back",
    "Trial 2:2back",
    "Trial 3:3back",
    "Trial 4:2back",
    "Trial 5:3back",
    "Trial 6:0back",
]

# Workload label per trial (counter-balanced order 0->2->3->2->3->0)
# binary: 0 = low workload (0-back), 1 = high workload (2-back & 3-back)
TRIAL_LABEL_BINARY = {
    "Trial 1:0back": 0,
    "Trial 2:2back": 1,
    "Trial 3:3back": 1,
    "Trial 4:2back": 1,
    "Trial 5:3back": 1,
    "Trial 6:0back": 0,
}

# 3-class: 0 = 0-back, 1 = 2-back, 2 = 3-back
TRIAL_LABEL_MULTICLASS = {
    "Trial 1:0back": 0,
    "Trial 2:2back": 1,
    "Trial 3:3back": 2,
    "Trial 4:2back": 1,
    "Trial 5:3back": 2,
    "Trial 6:0back": 0,
}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CNN_FILTERS = [64, 128]
CNN_KERNELS = [5, 3]
POOL_SIZE = 2

LSTM_UNITS = 64
LSTM_LAYERS = 2

ATTENTION_UNITS = 64

DENSE_UNITS = [64, 32]
DROPOUT = 0.3

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SUBJECT_FRACTION = 0.15
VAL_SUBJECT_FRACTION = 0.15
