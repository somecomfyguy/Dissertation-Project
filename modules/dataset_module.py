"""
Dataset processing module for both OAKBAT and Swinney datasets. Use 2 passes 
in order to optimize memory usage.

Pass 1: Stream files -> Compute spectrograms -> Accumulate normalizer stats
         (one file at a time, IQ discarded after spectrogram computation)
Pass 2: Stream files again -> Compute spectrograms -> normalize -> Save to disk
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat
from scipy.signal import stft
from skimage.transform import resize

# Global variables
SAMPLE_RATE = 5e6  # 5 MHz
oakbat_gps_filepath = Path("L1")
oakbat_gal_filepath = Path("E1")

# Remap of dataset categories for the Swinney dataset
SWINNEY_CLASS_MAP = {
    "NoJam":       "clean",
    "SingleAM":    "jam_am",
    "SingleChirp": "jam_chirp",
    "SingleFM":    "jam_fm",
    "DME":         "jam_dme",
    "NB":          "jam_narrowband",
}


@dataclass
class OakbatScenario:
    """Metadata for a single OAKBAT scenario file."""
    filename: str
    constellation: str
    scenario: str
    spoofing_class: str
    onset_time: float


SCENARIOS_GPS = [
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os1.bin"), "gps", "ds1",
                   "spoof_overpowered_instant", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os2.bin"), "gps", "ds2",
                   "spoof_overpowered_gradual", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os3.bin"), "gps", "ds3",
                   "spoof_matched_time", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os4.bin"), "gps", "ds4",
                   "spoof_matched_dynamic", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os5.bin"), "gps", "ds5",
                   "spoof_position_push", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gps_filepath, "os6.bin"), "gps", "ds6",
                   "spoof_dynamic_position", onset_time=120.0),
]

SCENARIOS_GALILEO = [
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os9a.bin"), "galileo", "ds1",
                   "spoof_overpowered_instant", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os10.bin"), "galileo", "ds2",
                   "spoof_overpowered_gradual", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os11.bin"), "galileo", "ds3",
                   "spoof_matched_time", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os12.bin"), "galileo", "ds4",
                   "spoof_matched_dynamic", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os13.bin"), "galileo", "ds5",
                   "spoof_position_push", onset_time=120.0),
    OakbatScenario(os.path.join(oakbat_gal_filepath, "os14.bin"), "galileo", "ds6",
                   "spoof_dynamic_position", onset_time=120.0),
]

ALL_SCENARIOS = SCENARIOS_GPS + SCENARIOS_GALILEO


# Segment metadata
@dataclass
class SegmentMeta:
    """
    Lightweight metadata for a segment. Does NOT hold IQ data — only enough
    information to re-read the segment from disk in pass 2.
    """
    label: str
    source_file: str        # Full path to .bin or .mat file
    constellation: str
    scenario: str
    start_sample: int       # Offset into the file (for OAKBAT)
    num_samples: int        # Window length in samples
    is_spoofed: bool
    dataset: str            # "oakbat" or "swinney"


# STFT configuration
@dataclass
class STFTParams:

    """STFT configuration."""
    nperseg: int = 256              # samples per segment
    noverlap: int = 192             # overlap between samples
    window: str = "hann"            # window
    output_size: tuple = (128, 128) # ouput size of spectrograms
    log_scale: bool = True          # log scale power better for CNNs (check this)
    epsilon: float = 1e-10          # epsilon


def compute_spectrogram(segment_data: np.ndarray, fs: float,
                        params: STFTParams = STFTParams()) -> np.ndarray:
                        
    """Compute STFT spectrogram from IQ segment and resize to output_size."""
    _, _, Zxx = stft(segment_data, fs=fs,
                     nperseg=params.nperseg,
                     noverlap=params.noverlap,
                     window=params.window)
    mag = np.abs(Zxx)
    if params.log_scale:
        mag = 10 * np.log10(mag ** 2 + params.epsilon)
    spectrogram = resize(mag, params.output_size,
                         anti_aliasing=True,
                         preserve_range=True).astype(np.float32)
    return spectrogram


class SpectrogramNormalizer:
    """Handles per-image or global (z-score) normalization."""

    def __init__(self, mode: str = "global"):
        self.mode = mode
        self._running_sum = 0.0
        self._running_sq_sum = 0.0
        self._count = 0
        self.mean = None
        self.std = None

    def update(self, spectrogram: np.ndarray):
        self._running_sum += spectrogram.sum()
        self._running_sq_sum += (spectrogram ** 2).sum()
        self._count += spectrogram.size

    def finalize(self):
        if self._count == 0:
            raise RuntimeError("No data accumulated. Call update() first.")
        self.mean = self._running_sum / self._count
        self.std = np.sqrt(self._running_sq_sum / self._count - self.mean ** 2)
        if self.std < 1e-8:
            self.std = 1.0
        print(f"[Normalizer] Global mean={self.mean:.4f}, std={self.std:.4f}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        if self.mode == "per_image":
            smin, smax = spectrogram.min(), spectrogram.max()
            if smax - smin < 1e-8:
                return np.zeros_like(spectrogram)
            return (spectrogram - smin) / (smax - smin)
        else:
            if self.mean is None:
                raise RuntimeError("Call finalize() before transform().")
            return (spectrogram - self.mean) / self.std

    def save(self, path: str):
        stats = {"mode": self.mode, "mean": float(self.mean), "std": float(self.std)}
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SpectrogramNormalizer":
        with open(path) as f:
            stats = json.load(f)
        norm = cls(mode=stats["mode"])
        norm.mean = stats["mean"]
        norm.std = stats["std"]
        return norm


def read_oakbat_chunk(filepath: str, start_sample: int,
                      num_samples: int) -> np.ndarray:
    """
    Read a specific chunk of IQ from an OAKBAT .bin file without
    loading the entire file. Seeks directly to the offset.
    """
    # Each complex sample = 2 × int16 = 4 bytes
    offset_bytes = start_sample * 2 * 2  # 2 components × 2 bytes each
    count = num_samples * 2              # I and Q interleaved

    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        raw = np.frombuffer(f.read(count * 2), dtype=np.int16)

    iq = raw.reshape(-1, 2)
    return iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)


def read_swinney_file(filepath: str) -> np.ndarray:
    """Read a single Swinney .mat file and return complex IQ."""
    mat_data = loadmat(filepath)
    mat_var_name = "GNSS_plus_Jammer_awgn"

    if mat_var_name not in mat_data:
        data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if len(data_keys) == 1:
            iq = mat_data[data_keys[0]].squeeze()
        else:
            raise KeyError(f"Variable '{mat_var_name}' not found. Keys: {data_keys}")
    else:
        iq = mat_data[mat_var_name].squeeze()

    if not np.iscomplexobj(iq):
        if iq.ndim == 2 and iq.shape[1] == 2:
            iq = iq[:, 0] + 1j * iq[:, 1]
        else:
            iq = iq.astype(np.complex64)
    return iq.astype(np.complex64)


def read_segment_iq(meta: SegmentMeta) -> np.ndarray:
    """Read the IQ data for a segment from disk, dispatching by dataset type."""
    if meta.dataset == "oakbat":
        return read_oakbat_chunk(meta.source_file, meta.start_sample,
                                 meta.num_samples)
    else:  # swinney
        return read_swinney_file(meta.source_file)


def scan_oakbat_segments(oakbat_dir: str,
                         constellations: list[str] = ["gps", "galileo"],
                         window_duration_s: float = 0.02,
                         overlap: float = 0.0,
                         max_file_duration_s: Optional[float] = None,
                         ) -> list[SegmentMeta]:
    """
    Scan OAKBAT files and return lightweight segment metadata.
    Files are NOT read — only their size is checked to determine
    how many segments they contain.
    """
    oakbat_path = Path(oakbat_dir)
    scenarios = [s for s in ALL_SCENARIOS if s.constellation in constellations]
    window_samples = int(window_duration_s * SAMPLE_RATE)
    hop_samples = int(window_samples * (1 - overlap))

    all_meta: list[SegmentMeta] = []

    for scenario in scenarios:
        filepath = oakbat_path / scenario.filename
        if not filepath.exists():
            pattern = f"*{scenario.constellation}*{scenario.scenario}*"
            candidates = list(oakbat_path.glob(pattern))
            if candidates:
                filepath = candidates[0]
            else:
                print(f"  [SKIP] File not found: {scenario.filename}")
                continue

        # Determine file length from file size (no data loaded)
        file_size_bytes = filepath.stat().st_size
        total_samples = file_size_bytes // 4  # 2 × int16 per complex sample

        if max_file_duration_s is not None:
            max_samples = int(max_file_duration_s * SAMPLE_RATE)
            total_samples = min(total_samples, max_samples)

        onset_sample = int(scenario.onset_time * SAMPLE_RATE)

        print(f"  Scanning {filepath.name} ({scenario.constellation} "
              f"{scenario.scenario}): {total_samples / SAMPLE_RATE:.1f}s, "
              f"onset at {scenario.onset_time}s")

        n_clean = 0
        n_spoofed = 0
        start = 0
        while start + window_samples <= total_samples:
            end = start + window_samples

            if end <= onset_sample:
                label = "clean"
                is_spoofed = False
                n_clean += 1
            elif start >= onset_sample:
                label = scenario.spoofing_class
                is_spoofed = True
                n_spoofed += 1
            else:
                start += hop_samples
                continue

            all_meta.append(SegmentMeta(
                label=label,
                source_file=str(filepath),
                constellation=scenario.constellation,
                scenario=scenario.scenario,
                start_sample=start,
                num_samples=window_samples,
                is_spoofed=is_spoofed,
                dataset="oakbat",
            ))
            start += hop_samples

        print(f"    → {n_clean + n_spoofed} segments "
              f"(clean: {n_clean}, spoofed: {n_spoofed})")

    return all_meta


def scan_swinney_segments(swinney_dir: str,
                          split: str = "training") -> list[SegmentMeta]:
    """Scan Swinney .mat files and return lightweight segment metadata."""
    class_map = SWINNEY_CLASS_MAP
    base_dir = Path(swinney_dir) / split.capitalize()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Swinney {split} directory not found: {base_dir}")

    all_meta: list[SegmentMeta] = []

    for class_dir_name, unified_label in class_map.items():
        class_dir = base_dir / class_dir_name
        if not class_dir.exists():
            print(f"  [WARN] Directory not found, skipping: {class_dir}")
            continue

        mat_files = sorted(class_dir.glob("*.mat"))
        if not mat_files:
            print(f"  [WARN] No .mat files in {class_dir}")
            continue

        for mat_path in mat_files:
            all_meta.append(SegmentMeta(
                label=unified_label,
                source_file=str(mat_path),
                constellation="gps",
                scenario=f"swinney_{split}",
                start_sample=0,
                num_samples=0,  # Full file for Swinney
                is_spoofed=False,
                dataset="swinney",
            ))

        print(f"  {unified_label} ({class_dir_name}): "
              f"{len(mat_files)} files found")

    return all_meta


def create_splits(segments: list[SegmentMeta],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  balance_classes: bool = True,
                  max_per_class: Optional[int] = None,
                  seed: int = 42) -> dict[str, list[SegmentMeta]]:
    """Create stratified train/val/test splits from segment metadata."""
    rng = np.random.default_rng(seed)

    by_label: dict[str, list[SegmentMeta]] = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)

    print("\n[Split] Class distribution before balancing:")
    for label, segs in sorted(by_label.items()):
        print(f"  {label}: {len(segs)} segments")

    if balance_classes:
        min_count = min(len(v) for v in by_label.values())
        if max_per_class is not None:
            min_count = min(min_count, max_per_class)
        for label in by_label:
            indices = rng.permutation(len(by_label[label]))[:min_count]
            by_label[label] = [by_label[label][i] for i in indices]
        print(f"[Split] Balanced to {min_count} samples per class")

    splits = {"train": [], "val": [], "test": []}
    for label, segs in by_label.items():
        perm = rng.permutation(len(segs))
        n = len(segs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for i, idx in enumerate(perm):
            if i < n_train:
                splits["train"].append(segs[idx])
            elif i < n_train + n_val:
                splits["val"].append(segs[idx])
            else:
                splits["test"].append(segs[idx])

    for split_name, split_segs in splits.items():
        labels = [s.label for s in split_segs]
        unique, counts = np.unique(labels, return_counts=True)
        dist = ", ".join(f"{l}: {c}" for l, c in zip(unique, counts))
        print(f"[Split] {split_name}: {len(split_segs)} total ({dist})")

    return splits


def save_dataset_streaming(
    splits: dict[str, list[SegmentMeta]],
    output_dir: str,
    fs: float,
    stft_params: STFTParams,
    normalizer: SpectrogramNormalizer,
    save_format: str = "npy",
):
    """
    Pass 2: Re-read IQ for each segment, compute spectrogram, normalize,
    and save to disk. Only one segment's IQ is in memory at a time.
    """
    out = Path(output_dir)
    metadata = {
        "fs": fs,
        "stft_params": {
            "nperseg": stft_params.nperseg,
            "noverlap": stft_params.noverlap,
            "window": stft_params.window,
            "output_size": list(stft_params.output_size),
            "log_scale": stft_params.log_scale,
        },
        "splits": {},
    }

    for split_name, segments in splits.items():
        split_dir = out / split_name
        counters: dict[str, int] = {}
        metadata["splits"][split_name] = {}

        for i, meta in enumerate(segments):
            iq = read_segment_iq(meta)
            spec = compute_spectrogram(iq, fs, stft_params)
            spec = normalizer.transform(spec)

            label_dir = split_dir / meta.label
            label_dir.mkdir(parents=True, exist_ok=True)

            idx = counters.get(meta.label, 0) + 1
            counters[meta.label] = idx

            if save_format == "npy":
                np.save(str(label_dir / f"spec_{idx:05d}.npy"), spec)
            elif save_format == "png":
                from PIL import Image
                img = ((spec - spec.min()) / (spec.max() - spec.min() + 1e-10) * 255)
                Image.fromarray(img.astype(np.uint8)).save(
                    str(label_dir / f"spec_{idx:05d}.png"))

            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] Saved {i + 1}/{len(segments)}")

        for label, count in counters.items():
            metadata["splits"][split_name][label] = count
        print(f"  [{split_name}] Done: {sum(counters.values())} spectrograms")

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    normalizer.save(str(out / "normalization_stats.json"))

    print(f"\n[Save] Dataset saved to {out}")


def compute_normalization_streaming(
    segments: list[SegmentMeta],
    fs: float,
    stft_params: STFTParams,
) -> SpectrogramNormalizer:
    """
    Pass 1: Stream through segments computing spectrograms to accumulate
    global normalization stats. Only one segment's IQ in memory at a time.
    """
    normalizer = SpectrogramNormalizer(mode="global")
    total = len(segments)

    print(f"\n[Norm] Computing global stats from {total} training segments...")
    for i, meta in enumerate(segments):
        iq = read_segment_iq(meta)
        spec = compute_spectrogram(iq, fs, stft_params)
        normalizer.update(spec)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total}")

    normalizer.finalize()
    return normalizer


def run_oakbat_pipeline(
    oakbat_dir: str,
    output_dir: str,
    constellations: list[str] = ["gps", "galileo"],
    window_duration_s: float = 0.02,
    overlap: float = 0.0,
    stft_params: STFTParams = STFTParams(),
    normalization_mode: str = "global",
    balance: bool = True,
    max_per_class: Optional[int] = None,
    save_format: str = "npy",
    max_file_duration_s: Optional[float] = None,
    external_norm_path: Optional[str] = None,
):
    """
    Memory-optimized OAKBAT pipeline.

    Peak memory: ~one window of IQ data + one spectrogram at a time.
    """
    print("=" * 60)
    print("OAKBAT Processing Pipeline (streaming)")
    print("=" * 60)
    print(f"Source:         {oakbat_dir}")
    print(f"Constellations: {constellations}")
    print(f"Window:         {window_duration_s * 1000:.0f} ms")
    print(f"STFT:           nperseg={stft_params.nperseg}, "
          f"noverlap={stft_params.noverlap}")
    print(f"Output size:    {stft_params.output_size}")
    print(f"Normalization:  {normalization_mode}")
    print()

    # --- Scan: collect metadata only ---
    print("[Phase 1] Scanning files for segment metadata...")
    all_meta = scan_oakbat_segments(
        oakbat_dir, constellations, window_duration_s, overlap,
        max_file_duration_s)

    if not all_meta:
        print("\n[ERROR] No segments found. Check file paths.")
        return

    print(f"\nTotal segments found: {len(all_meta)}")

    # --- Split ---
    splits = create_splits(all_meta, balance_classes=balance,
                           max_per_class=max_per_class)

    # --- Normalize ---
    if external_norm_path:
        print(f"\n[Norm] Loading external normalization from {external_norm_path}")
        normalizer = SpectrogramNormalizer.load(external_norm_path)
        print(f"[Norm] mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    else:
        print("\n[Phase 2] Computing normalization stats (streaming)...")
        normalizer = compute_normalization_streaming(
            splits["train"], SAMPLE_RATE, stft_params)

    # --- Save ---
    print("\n[Phase 3] Saving spectrograms (streaming)...")
    save_dataset_streaming(splits, output_dir, SAMPLE_RATE, stft_params,
                           normalizer, save_format)

    print("\n" + "=" * 60)
    print("OAKBAT pipeline complete!")
    print("=" * 60)


def run_swinney_pipeline(
    swinney_dir: str,
    output_dir: str,
    stft_params: STFTParams = STFTParams(),
    normalization_mode: str = "global",
    external_norm_path: Optional[str] = None,
    balance: bool = True,
    max_per_class: Optional[int] = None,
    save_format: str = "npy",
    use_original_splits: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stats_only: bool = False,
):
    """
    Memory-optimized Swinney pipeline.

    Swinney files are small (~50K samples each), so memory is less of a
    concern here, but we still use the streaming approach for consistency.
    """
    print("=" * 60)
    print("Swinney Processing Pipeline (streaming)")
    print("=" * 60)
    print(f"Source:         {swinney_dir}")
    print(f"STFT:           nperseg={stft_params.nperseg}, "
          f"noverlap={stft_params.noverlap}")
    print(f"Output size:    {stft_params.output_size}")
    print(f"Normalization:  {normalization_mode}"
          f"{' (external)' if external_norm_path else ''}")
    print()

    # --- Scan ---
    if use_original_splits:
        print("Scanning training set...")
        train_meta = scan_swinney_segments(swinney_dir, "training")
        print(f"  Total training: {len(train_meta)}")

        print("\nScanning testing set...")
        test_meta = scan_swinney_segments(swinney_dir, "testing")
        print(f"  Total testing: {len(test_meta)}")

        if not train_meta:
            print("\n[ERROR] No training segments found.")
            return

        # Balance and carve out validation from training
        rng = np.random.default_rng(42)
        by_label: dict[str, list[SegmentMeta]] = {}
        for seg in train_meta:
            by_label.setdefault(seg.label, []).append(seg)

        if balance:
            min_count = min(len(v) for v in by_label.values())
            if max_per_class is not None:
                min_count = min(min_count, max_per_class)
            for label in by_label:
                indices = rng.permutation(len(by_label[label]))[:min_count]
                by_label[label] = [by_label[label][i] for i in indices]
            print(f"\n[Split] Balanced training to {min_count} per class")

        # Balance test set
        if balance:
            test_by_label: dict[str, list[SegmentMeta]] = {}
            for seg in test_meta:
                test_by_label.setdefault(seg.label, []).append(seg)
            test_min = min(len(v) for v in test_by_label.values())
            if max_per_class is not None:
                test_min = min(test_min, max_per_class)
            balanced_test = []
            for label in test_by_label:
                indices = rng.permutation(len(test_by_label[label]))[:test_min]
                balanced_test.extend(test_by_label[label][i] for i in indices)
            test_meta = balanced_test

        val_frac = val_ratio / (1.0 - (1.0 - train_ratio - val_ratio))
        splits = {"train": [], "val": [], "test": test_meta}

        for label, segs in by_label.items():
            perm = rng.permutation(len(segs))
            n_val = int(len(segs) * val_frac)
            for i, idx in enumerate(perm):
                if i < n_val:
                    splits["val"].append(segs[idx])
                else:
                    splits["train"].append(segs[idx])

        for split_name, split_segs in splits.items():
            labels = [s.label for s in split_segs]
            unique, counts = np.unique(labels, return_counts=True)
            dist = ", ".join(f"{l}: {c}" for l, c in zip(unique, counts))
            print(f"[Split] {split_name}: {len(split_segs)} total ({dist})")
    else:
        print("Scanning all segments (pooled)...")
        all_meta = []
        for split in ["training", "testing"]:
            try:
                all_meta.extend(scan_swinney_segments(swinney_dir, split))
            except FileNotFoundError:
                print(f"  [Note] No {split} directory found")
        splits = create_splits(all_meta, train_ratio=train_ratio,
                               val_ratio=val_ratio, balance_classes=balance,
                               max_per_class=max_per_class)

    # --- Normalize ---
    if external_norm_path:
        print(f"\n[Norm] Loading external normalization from {external_norm_path}")
        normalizer = SpectrogramNormalizer.load(external_norm_path)
        print(f"[Norm] mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")
    else:
        normalizer = compute_normalization_streaming(
            splits["train"], SAMPLE_RATE, stft_params)

    if stats_only:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        normalizer.save(str(out_path / "normalization_stats.json"))
        print(f"\n[Stats] Saved to {out_path / 'normalization_stats.json'}")
        return normalizer

    # --- Save ---
    print("\n[Phase 3] Saving spectrograms (streaming)...")
    save_dataset_streaming(splits, output_dir, SAMPLE_RATE, stft_params,
                           normalizer, save_format)

    print("\n" + "=" * 60)
    print("Swinney pipeline complete!")
    print("=" * 60)


def compute_joint_normalization(
    oakbat_train_meta: list[SegmentMeta],
    swinney_train_meta: list[SegmentMeta],
    stft_params: STFTParams,
    fs: float = SAMPLE_RATE,
    output_path: Optional[str] = None,
) -> SpectrogramNormalizer:
    """
    Compute unified normalization stats across both datasets' training
    segments using streaming (no IQ accumulation).
    """
    combined = oakbat_train_meta + swinney_train_meta
    print(f"[JointNorm] Computing stats across {len(combined)} segments "
          f"({len(oakbat_train_meta)} OAKBAT + {len(swinney_train_meta)} Swinney)")

    normalizer = compute_normalization_streaming(combined, fs, stft_params)

    if output_path:
        normalizer.save(output_path)
        print(f"[JointNorm] Stats saved to {output_path}")

    return normalizer
