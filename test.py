import numpy as np
from pathlib import Path

spec_dir = Path("./Output/combined_spectrograms")
nan_count = 0
inf_count = 0
for feat_file in spec_dir.rglob("feat_*.npy"):
    feat = np.load(feat_file)
    if np.any(np.isnan(feat)):
        nan_count += 1
        print(f"  NaN in {feat_file}")
    if np.any(np.isinf(feat)):
        inf_count += 1
        print(f"  Inf in {feat_file}")
print(f"\nTotal: {nan_count} NaN files, {inf_count} Inf files")