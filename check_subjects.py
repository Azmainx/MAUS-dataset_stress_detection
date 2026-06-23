"""
check_subjects.py — verify that all subjects are discovered correctly.

Run this script to confirm all valid subject folders are found before running the full pipeline.
"""

import config as cfg
from dataset_builder import list_subject_dirs


if __name__ == "__main__":
    print(f"Scanning: {cfg.RAW_DATA_DIR}")
    subs = list_subject_dirs(cfg.RAW_DATA_DIR)
    
    if not subs:
        print("ERROR: No valid subject folders found!")
        exit(1)
    
    print(f"\nFound {len(subs)} valid subject folders:")
    for i, s in enumerate(subs, 1):
        # Extract subject ID (folder name) from path
        subject_id = s.split("\\")[-1]
        print(f"  {i:2d}. {subject_id}  ({s})")
    
    print(f"\n✓ All subjects ready for pipeline.")
