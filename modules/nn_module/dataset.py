"""
PyTorch Dataset for pre-computed spectrogram .npy files.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """
    Loads pre-computed .npy spectrograms from a split directory.

    Each spectrogram is a 128x128 float32 array (single-channel).
    For models expecting 3-channel input (e.g. ResNet-18), the single
    channel is replicated across RGB.

    Directory layout expected:
        split_dir/
            clean/         spec_00001.npy …
            jam_am/        spec_00001.npy …
            …
    """

    def __init__(self, split_dir: str, class_to_idx: dict = None,
                 num_channels: int = 3):
        """
        Args:
            split_dir:     Path to train/, val/, or test/ directory.
            class_to_idx:  Optional pre-defined label mapping. If None,
                           inferred from subdirectory names (sorted).
            num_channels:  Number of channels to expand to (default 3
                           for ImageNet-pretrained backbones).
        """
        self.num_channels = num_channels
        self.samples: list[tuple[str, int]] = []
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
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.samples.append((str(npy_file), idx))

        print(f"  Loaded {len(self.samples)} samples, "
              f"{self.num_classes} classes from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        spec = np.load(filepath).astype(np.float32)
        tensor = torch.from_numpy(spec).unsqueeze(0)
        if self.num_channels > 1:
            tensor = tensor.expand(self.num_channels, -1, -1)
        return tensor, label