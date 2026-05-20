"""
PyTorch Dataset for pre-computed spectrogram (and optional feature) .npy files.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from modules.nn_module.augmentation import SpectrogramAugmentor


class SpectrogramDataset(Dataset):
    """
    Loads pre-computed .npy spectrograms from a split directory.

    Each spectrogram is a 128x128 float32 array (single-channel).
    For models expecting 3-channel input (e.g. ResNet-18), the single
    channel is replicated across RGB.

    When load_features=True, expects parallel files:
        split_dir/class_name/spec_00001.npy
        split_dir/class_name/feat_00001.npy

    When augment=True, applies random spectrogram augmentations during
    loading (training only — do NOT enable for val/test).

    Returns:
        If load_features=False: (spectrogram_tensor, label)
        If load_features=True:  ((spectrogram_tensor, feature_tensor), label)
    """

    def __init__(self, split_dir: str, class_to_idx: dict = None,
                 num_channels: int = 3, load_features: bool = False,
                 augment: bool = False):
        """
        Args:
            split_dir:      Path to train/, val/, or test/ directory.
            class_to_idx:   Optional pre-defined label mapping.
            num_channels:   Channels to expand spectrogram to (default 3).
            load_features:  If True, also load feat_NNNNN.npy files.
            augment:        If True, apply random augmentations (train only).
        """
        self.num_channels  = num_channels
        self.load_features = load_features
        self.augment       = augment
        self.augmentor     = SpectrogramAugmentor() if augment else None
        self.samples: list[tuple[str, str | None, int]] = []
        split_path = Path(split_dir)

        class_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])

        if class_to_idx is None:
            self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes   = len(self.class_to_idx)

        for class_dir in class_dirs:
            label = class_dir.name
            if label not in self.class_to_idx:
                print(f"  [WARN] Skipping unknown class dir: {label}")
                continue
            idx = self.class_to_idx[label]
            for spec_file in sorted(class_dir.glob("spec_*.npy")):
                feat_file = None
                if load_features:
                    feat_name = spec_file.name.replace("spec_", "feat_")
                    feat_path = class_dir / feat_name
                    if feat_path.exists():
                        feat_file = str(feat_path)
                    else:
                        print(f"  [WARN] Missing feature file: {feat_path}")
                        continue
                self.samples.append((str(spec_file), feat_file, idx))

        aug_status  = " + augmentation" if augment else ""
        feat_status = " + features" if load_features else ""
        print(f"  Loaded {len(self.samples)} samples{feat_status}{aug_status}, "
              f"{self.num_classes} classes from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        spec_path, feat_path, label = self.samples[index]

        spec = np.load(spec_path).astype(np.float32)

        # Apply augmentation before channel expansion (train only)
        if self.augment and self.augmentor is not None:
            spec = self.augmentor(spec)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        if self.num_channels > 1:
            spec_tensor = spec_tensor.expand(self.num_channels, -1, -1)

        if self.load_features and feat_path is not None:
            feat = np.load(feat_path).astype(np.float32)
            feat_tensor = torch.from_numpy(feat)
            return (spec_tensor, feat_tensor), label

        return spec_tensor, label