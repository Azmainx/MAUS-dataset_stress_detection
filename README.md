# MAUS Mental-Workload Pipeline: ECG + PPG + GSR → 1D-CNN → BiLSTM → Attention → Dense → SHAP

This implements the pipeline you sketched, tailored to the **actual structure of the MAUS
dataset** (confirmed from the documentation + the sample subject's files you uploaded).

---

## 0. What the data actually looks like (1 subject = 1 folder)

| File | Shape | Meaning |
|---|---|---|
| `inf_ecg.csv`, `inf_ppg.csv`, `inf_gsr.csv` | 76800 rows × 6 cols (`Trial 1:0back ... Trial 6:0back`) | Raw 256 Hz Procomp-Infiniti signals, one column per trial, 5 min/trial → 300 s × 256 Hz = 76800 samples |
| `inf_resting.csv` | 74970 rows × 3 cols (`Resting_ECG/PPG/GSR`) | Resting baseline, ~293 s @ 256 Hz |
| `pixart.csv`, `pixart_resting.csv` | wrist-PPG @ 100 Hz | Not used by the CNN (we use the Infiniti fingertip channels: ECG, PPG, GSR — the 3 modalities you specified) |
| `trial_#_##.csv`, `rest_#.csv` | IBI/PPI tables | Already-derived HRV features — optional auxiliary features, not needed for the raw-signal CNN |
| `record.xlsx` | event timestamps + N-back responses | Not required for the workload label (label comes from trial design) |
| `NASA_TLX.csv`, `PSQI.csv` | subjective workload / sleep quality | Optional regression targets / covariates |

**Label comes from the experiment design, not from a column**: trials are run in the fixed
counter-balanced order **0→2→3→2→3→0**, i.e.

```
Trial 1 = 0-back  → LOW workload   → label 0
Trial 2 = 2-back  → HIGH workload  → label 1
Trial 3 = 3-back  → HIGH workload  → label 1
Trial 4 = 2-back  → HIGH workload  → label 1
Trial 5 = 3-back  → HIGH workload  → label 1
Trial 6 = 0-back  → LOW workload   → label 0
```
(For a 3-class version: 0-back=0, 2-back=1, 3-back=2 — `config.py` has a flag for this.)

There are **22 subject folders** in the full MAUS release; you've supplied 1 as a template.
The code below is written to loop over `data/subject_XX/...` automatically, so dropping in
the other 21 folders (same filenames) is all that's needed to scale up.

---

## 1. Pipeline overview (matches your diagram)

```
MAUS raw CSVs (ECG, PPG, GSR @ 256Hz, per subject)
        │
        ▼
┌────────────────────┐
│ Preprocessing       │  Filtering (bandpass/notch), Normalization (z-score, per-subject),
│                      │  Windowing (10s windows, 50% overlap) + label assignment
└────────┬─────────────┘
         ▼
┌────────────────────┐
│ 1D-CNN (2 layers)    │  Conv1D k=5,f=64 → BN → ReLU → MaxPool
│                      │  Conv1D k=3,f=128 → BN → ReLU → MaxPool
└────────┬─────────────┘
         ▼
┌────────────────────┐
│ BiLSTM (2 layers)    │  Bi-LSTM(64, return_seq) → Bi-LSTM(64, return_seq)
└────────┬─────────────┘
         ▼
┌────────────────────┐
│ Self-Attention       │  Additive (Bahdanau-style) attention over the BiLSTM time axis
└────────┬─────────────┘
         ▼
┌────────────────────┐
│ Dense + Dropout      │  64 → Dropout(0.3) → 32 → Dropout(0.3) → Output (sigmoid/softmax)
└────────┬─────────────┘
         ▼
┌────────────────────┐
│ SHAP (Gradient)      │  Which time-steps / channels (ECG vs PPG vs GSR) drove the decision
└────────────────────┘
```

---

## 2. Files in this project

```
maus_pipeline/
├── config.py            # all hyperparameters & paths in one place
├── preprocessing.py      # filtering, normalization, windowing, labeling
├── dataset_builder.py    # loops over subjects, builds train/val/test tensors
├── model.py              # CNN-BiLSTM-Attention Keras model
├── train.py              # training loop, callbacks, subject-wise split
├── evaluate.py           # metrics, confusion matrix, ROC
├── shap_analysis.py      # GradientExplainer + plots
└── run_pipeline.py       # one entrypoint that runs everything end-to-end
```

## 3. Step-by-step explanation

### Step 1 — Preprocessing (`preprocessing.py`)
1. **Load** the 3 raw channels for a subject (`inf_ecg.csv`, `inf_ppg.csv`, `inf_gsr.csv`), trial by trial (one of the 6 columns at a time).
2. **Filtering**
   - ECG: Butterworth band-pass 0.5–40 Hz (removes baseline wander + EMG noise) + optional 50/60 Hz notch.
   - PPG: Butterworth band-pass 0.5–8 Hz (PPG signal band).
   - GSR: low-pass 1 Hz (GSR is slow-varying; tonic+phasic skin conductance).
3. **Normalization**: per-subject, per-channel z-score (`(x-μ)/σ`) computed on that subject's resting + trial data, so the network learns workload-relevant shape, not absolute amplitude (which varies a lot between people/sessions).
4. **Windowing**: slide a 10 s window (2560 samples @256Hz) with 50% overlap (stride 5s) across each trial's filtered signal. Each window gets the trial's workload label. This turns 6 trials × ~300s into hundreds of labeled (2560,3) windows per subject — this is what makes a deep model feasible with limited subjects.
5. Output: `X` shape `(n_windows, 2560, 3)` — channels = [ECG, PPG, GSR]; `y` shape `(n_windows,)`.

### Step 2 — Dataset assembly (`dataset_builder.py`)
- Loops over every subject folder, applies Step 1, concatenates all windows.
- **Subject-wise train/val/test split** (not random window split!) — e.g. 16 subjects train / 3 val / 3 test — to avoid leakage (windows from the same subject/trial are highly correlated, random splitting would let the model "memorize" a subject's baseline and inflate accuracy).
- Optionally down-samples the windows (2560 → e.g. 512 via decimation) to keep the CNN+BiLSTM tractable; configurable in `config.py`.

### Step 3 — Model (`model.py`)
- **1D-CNN block** (2 layers, kernels 3 & 5 as you specified, 64→128 filters): extracts local morphological features (QRS shape, PPG pulse shape, GSR slope changes) and downsamples the time axis via MaxPooling.
- **BiLSTM block** (2 layers, 64 units): models temporal dependencies in both directions over the CNN's feature-map sequence — captures things like heart-rate variability trends across the window.
- **Attention layer**: learns a weight per time-step of the BiLSTM output and produces a single context vector — lets the model "point to" the most workload-discriminative segment of the window (and gives us something SHAP can explain).
- **Dense+Dropout head**: 64→32 with Dropout(0.3) each, final `sigmoid` (binary low/high workload) or `softmax` (3-class 0/2/3-back).

### Step 4 — Training (`train.py`)
- Loss: binary (or categorical) cross-entropy; Adam optimizer, lr=1e-3 with ReduceLROnPlateau.
- Class weighting (4 high-workload trials vs 2 low-workload trials → mild imbalance).
- EarlyStopping on val loss, ModelCheckpoint saves best model.
- 5-fold subject-wise cross-validation recommended for the final reported metric (script supports both single split and CV).

### Step 5 — Evaluation (`evaluate.py`)
- Accuracy, F1, ROC-AUC, confusion matrix at the **window level**, then majority-vote aggregation to a **trial-level** decision (since the real use-case is "what is this person's workload during this trial", not single 10s windows).

### Step 6 — SHAP analysis (`shap_analysis.py`)
- Uses `shap.GradientExplainer` on the trained Keras model with a background sample of training windows.
- Produces:
  1. **Per-channel importance** — sum |SHAP| over time for ECG vs PPG vs GSR → which modality matters most for workload detection.
  2. **Per-time-step heatmap** — overlaid on the raw waveform, showing which part of the 10s window (e.g. around an R-peak, or a GSR rise) drove the prediction.
  3. **Summary plot** across many windows.

---

## 4. How to run

```bash
pip install tensorflow scikit-learn shap pandas numpy scipy matplotlib --break-system-packages

# 1. Put all 22 subject folders under ./data/subject_01 ... subject_22
#    each containing inf_ecg.csv, inf_ppg.csv, inf_gsr.csv (and optionally inf_resting.csv)

python run_pipeline.py --data_dir ./data --epochs 50 --task binary
```

This will preprocess all subjects, train the CNN-BiLSTM-Attention model, evaluate it, and
save SHAP plots to `./outputs/`.

With only the **one sample subject** you've attached, `run_pipeline.py` will still run
end-to-end (it's a good smoke test) but obviously can't subject-wise-split or generalize —
you need the other 21 subject folders for a meaningful train/test result.
#   M A U S - d a t a s e t _ s t r e s s _ d e t e c t i o n  
 