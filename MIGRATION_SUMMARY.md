# MAUS Pipeline Migration to Real Folder Layout

## Summary

The `maus_pipeline` has been successfully adapted to read directly from the real MAUS folder structure on disk at `E:\Thesis\MAUS` without requiring any file copying or manual reorganization.

## Changes Made

### 1. **config.py** — Updated paths configuration

Added new configuration variables:
- `MAUS_ROOT = r"E:\Thesis\MAUS"` — Root dataset directory
- `RAW_DATA_DIR = os.path.join(MAUS_ROOT, "Data", "Raw_data")` — Raw data directory
- `IBI_SEQUENCE_DIR = os.path.join(MAUS_ROOT, "Data", "IBI_sequence")` — For future use
- `SUBJECTIVE_RATING_DIR = os.path.join(MAUS_ROOT, "Subjective_rating")` — For future use
- `DATA_DIR = RAW_DATA_DIR` — Backward-compatible alias for CLI `--data_dir` argument

All paths use `os.path.join()` for cross-platform compatibility.

### 2. **dataset_builder.py** — Dynamic subject discovery

Rewrote `list_subject_dirs()` to:
- List all directories under `data_dir`
- **Keep only those that contain ALL three required CSV files**:
  - `inf_ecg.csv`
  - `inf_ppg.csv`
  - `inf_gsr.csv`
- Return full subject directory paths in sorted order (stable/reproducible)
- Removed old fallback for single sample subject at data_dir root
- Updated warning logic to only warn when exactly 1 subject is found

Result: Automatically discovers all valid subjects from the real folder structure.

### 3. **train.py** — Updated CLI default

Changed:
```python
p.add_argument("--data_dir", default=cfg.RAW_DATA_DIR)
```

Now defaults to the real MAUS data directory instead of `./data`.

### 4. **run_pipeline.py** — Updated CLI default

Changed:
```python
p.add_argument("--data_dir", default=cfg.RAW_DATA_DIR)
```

Now defaults to the real MAUS data directory instead of `./data`.

### 5. **check_subjects.py** — New verification script

Created a sanity-check utility that:
- Lists all discovered valid subject folders
- Verifies the correct number are found (22 expected)
- Shows subject IDs and full paths

Usage:
```bash
python check_subjects.py
```

## Verification Results

✅ **All 22 subjects discovered correctly**:
- Subjects: 002, 003, 004, 005, 006, 008, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019, 020, 021, 022, 023, 024, 025
- Missing: 007, 009 (as expected in the real dataset)
- All subjects contain the 3 required CSV files (inf_ecg.csv, inf_ppg.csv, inf_gsr.csv)

✅ **Dataset builder works correctly**:
- Builds dataset with proper subject-wise train/val/test split
- No file copying required
- Reads directly from `E:\Thesis\MAUS\Data\Raw_data`

✅ **Pipeline integration verified**:
- Training completes successfully with 1 epoch
- Model architecture builds correctly
- Dataset loaded properly with all subjects

## Usage

### 1. Verify subjects are discovered:
```bash
python check_subjects.py
```

### 2. Run full pipeline (with default real MAUS paths):
```bash
python run_pipeline.py --epochs 50
```

### 3. Run training only:
```bash
python train.py --epochs 50
```

### 4. Override data directory (if needed):
```bash
python run_pipeline.py --data_dir "E:\Thesis\MAUS\Data\Raw_data" --epochs 50
```

## Backward Compatibility

- CLI `--data_dir` argument still works
- Default now points to real MAUS directory
- `config.DATA_DIR` is maintained as alias for `config.RAW_DATA_DIR`
- Old `./data` layout no longer needed

## Files Modified

- ✏️ `config.py` — Added MAUS root paths
- ✏️ `dataset_builder.py` — Updated subject discovery logic
- ✏️ `train.py` — Updated CLI default
- ✏️ `run_pipeline.py` — Updated CLI default
- ✨ `check_subjects.py` — New verification script (created)
- ✓ `preprocessing.py` — No changes needed
- ✓ Other files — No changes needed

## Notes

- No file copying/moving required
- Code reads directly from `E:\Thesis\MAUS` in place
- Subject-wise splits ensure no data leakage across train/val/test
- Future work: Can add hooks to use IBI_sequence and Subjective_rating data when needed
