"""
Re-prepare TEXBAT spectrograms using per-image min-max normalisation
instead of the saved global z-score normaliser.

This removes all absolute power dependency — each spectrogram is
independently scaled to [0, 1]. If cross-dataset accuracy improves,
it confirms the global normaliser is a bottleneck for domain transfer.
"""
import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from modules.common.types import STFTParams, SAMPLE_RATE
from modules.common.spectrogram import compute_spectrogram
from modules.common.features import compute_features, FeatureNormalizer
from modules.dataset_module.process_texbat import scan_texbat_segments
from modules.dataset_module.dataset_main import read_segment_iq


TEXBAT_DIR      = "./modules/dataset_module/datasets/TexbatSpoofing"
TRAINED_NORM_DIR = "./Output/combined_spectrograms"
OUTPUT_DIR      = "./Output/texbat_spectrograms_perimage"


def perimage_minmax(spec: np.ndarray) -> np.ndarray:
    """Scale a single spectrogram to [0, 1] based on its own min/max."""
    lo, hi = spec.min(), spec.max()
    return ((spec - lo) / (hi - lo + 1e-10)).astype(np.float32)


def main():
    stft_params = STFTParams()

    # Still use the saved feature normaliser — per-image only affects
    # spectrograms. Features are a separate experiment.
    feat_norm = FeatureNormalizer.load(
        os.path.join(TRAINED_NORM_DIR, "feature_norm_stats.json"))

    print("[Prep] Scanning TEXBAT...")
    segments = scan_texbat_segments(TEXBAT_DIR)

    output_path = Path(OUTPUT_DIR)
    counters: dict[str, int] = {}

    print(f"[Prep] Processing {len(segments)} segments with per-image "
          f"min-max normalisation...")
    for i, meta in enumerate(segments):
        iq   = read_segment_iq(meta)
        spec = compute_spectrogram(iq, SAMPLE_RATE, stft_params)
        spec = perimage_minmax(spec)
        feat = compute_features(iq, SAMPLE_RATE)
        feat = feat_norm.transform(feat)

        label_dir = output_path / meta.label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(meta.label, 0) + 1
        counters[meta.label] = idx

        np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
        np.save(str(label_dir / f"feat_{idx:05d}.npy"), feat)

        if (i + 1) % 2000 == 0:
            print(f"  [{i+1}/{len(segments)}]")

    metadata = {
        "fs": SAMPLE_RATE,
        "source_dataset": "texbat",
        "normalisation": "per_image_minmax",
        "num_segments": sum(counters.values()),
        "per_class": counters,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Prep] Done. Per-class: {counters}")
    print(f"       Output: {output_path}")


if __name__ == "__main__":
    main()