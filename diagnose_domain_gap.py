"""
Diagnostic: compare OAKBAT (training) vs TEXBAT (eval) spectrogram
distributions to localise where the domain gap appears.

Dumps three things:
    1. Raw IQ magnitude statistics (pre-STFT).
    2. Spectrogram statistics before AND after the saved normaliser is
       applied — tells us whether the normaliser is the problem or the
       raw signal is.
    3. Side-by-side visual comparison of one clean segment from each
       dataset, at three stages: raw magnitude spectrum, log-magnitude
       STFT, and post-normalisation STFT.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.common.types import STFTParams, SAMPLE_RATE
from modules.common.spectrogram import compute_spectrogram, SpectrogramNormalizer
from modules.common.features import compute_features, FeatureNormalizer
from modules.dataset_module.process_oakbat import scan_oakbat_segments, read_oakbat_chunk
from modules.dataset_module.process_texbat import scan_texbat_segments, read_texbat_chunk


OAKBAT_DIR_1    = "C:/Users/Ciprian/Documents/temp"
OAKBAT_DIR_2    = "./modules/dataset_module/datasets/OakbatSpoofing"
TEXBAT_DIR      = "./modules/dataset_module/datasets/TexbatSpoofing"
NORM_DIR        = "./Output/combined_spectrograms"
OUTPUT_DIR      = "./Output/domain_gap_diagnostic"
N_SEGMENTS      = 100   # per dataset, all labelled 'clean'


def summarise(name: str, arr: np.ndarray) -> dict:
    return {
        "dataset": name,
        "mean":    float(np.mean(arr)),
        "std":     float(np.std(arr)),
        "min":     float(np.min(arr)),
        "max":     float(np.max(arr)),
        "p01":     float(np.percentile(arr, 1)),
        "p99":     float(np.percentile(arr, 99)),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stft_params = STFTParams()
    spec_norm   = SpectrogramNormalizer.load(
        os.path.join(NORM_DIR, "normalization_stats.json"))
    feat_norm   = FeatureNormalizer.load(
        os.path.join(NORM_DIR, "feature_norm_stats.json"))

    print(f"[Diag] Saved spec normaliser: mean={spec_norm.mean:.4f}, "
          f"std={spec_norm.std:.4f}")
    print(f"[Diag] Saved feat normaliser mean={feat_norm.mean.round(3)}")

    # Collect clean segments from both datasets
    print("\n[Diag] Scanning OAKBAT...")
    oak_1 = scan_oakbat_segments(OAKBAT_DIR_1)
    oak_2 = scan_oakbat_segments(OAKBAT_DIR_2)
    oak_all = oak_1 
    oak_clean = [s for s in oak_all if s.label == "clean"][:N_SEGMENTS]

    print("\n[Diag] Scanning TEXBAT...")
    tex_all   = scan_texbat_segments(TEXBAT_DIR)
    tex_clean = [s for s in tex_all if s.label == "clean"][:N_SEGMENTS]

    results = {}

    for name, segs, reader in [
        ("oakbat", oak_clean, read_oakbat_chunk),
        ("texbat", tex_clean, read_texbat_chunk),
    ]:
        print(f"\n[Diag] Processing {N_SEGMENTS} clean {name} segments...")
        iq_mags         = []
        raw_specs       = []
        normed_specs    = []
        raw_features    = []
        normed_features = []

        for seg in segs:
            iq = reader(seg.source_file, seg.start_sample, seg.num_samples)
            iq_mags.append(np.abs(iq))

            spec_raw = compute_spectrogram(iq, SAMPLE_RATE, stft_params)
            raw_specs.append(spec_raw)

            spec_normed = spec_norm.transform(spec_raw)
            normed_specs.append(spec_normed)

            feat_raw = compute_features(iq, SAMPLE_RATE)
            raw_features.append(feat_raw)
            normed_features.append(feat_norm.transform(feat_raw))

        iq_mags         = np.concatenate(iq_mags)
        raw_specs       = np.stack(raw_specs)
        normed_specs    = np.stack(normed_specs)
        raw_features    = np.stack(raw_features)
        normed_features = np.stack(normed_features)

        results[name] = {
            "iq_magnitude":           summarise(name, iq_mags),
            "spectrogram_raw":        summarise(name, raw_specs),
            "spectrogram_normalised": summarise(name, normed_specs),
            "features_raw_mean":      raw_features.mean(axis=0).tolist(),
            "features_raw_std":       raw_features.std(axis=0).tolist(),
            "features_normed_mean":   normed_features.mean(axis=0).tolist(),
            "features_normed_std":    normed_features.std(axis=0).tolist(),
        }

    # Write JSON summary
    out_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Diag] Stats saved to {out_path}")

    # Print the critical numbers inline so they show up in the terminal log
    print("\n" + "="*70)
    print(f"{'Quantity':<30} {'OAKBAT':>18} {'TEXBAT':>18}")
    print("="*70)
    for key in ["iq_magnitude", "spectrogram_raw", "spectrogram_normalised"]:
        for stat in ["mean", "std", "p01", "p99"]:
            ov = results["oakbat"][key][stat]
            tv = results["texbat"][key][stat]
            print(f"{key + '.' + stat:<30} {ov:>18.4f} {tv:>18.4f}")
        print("-"*70)

    # Visual: one segment from each, 3 stages
    oak_iq = read_oakbat_chunk(oak_clean[0].source_file,
                               oak_clean[0].start_sample,
                               oak_clean[0].num_samples)
    tex_iq = read_texbat_chunk(tex_clean[0].source_file,
                               tex_clean[0].start_sample,
                               tex_clean[0].num_samples)

    oak_raw    = compute_spectrogram(oak_iq, SAMPLE_RATE, stft_params)
    tex_raw    = compute_spectrogram(tex_iq, SAMPLE_RATE, stft_params)
    oak_normed = spec_norm.transform(oak_raw)
    tex_normed = spec_norm.transform(tex_raw)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (name, iq, spec_raw, spec_normed) in enumerate([
        ("OAKBAT clean", oak_iq, oak_raw, oak_normed),
        ("TEXBAT clean", tex_iq, tex_raw, tex_normed),
    ]):
        # Averaged PSD
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1/SAMPLE_RATE)) / 1e6
        psd   = np.fft.fftshift(np.abs(np.fft.fft(iq))**2) / len(iq)
        axes[row, 0].plot(freqs, 10*np.log10(psd + 1e-10))
        axes[row, 0].set_title(f"{name}: PSD (MHz)")
        axes[row, 0].set_xlabel("Frequency (MHz)")
        axes[row, 0].set_ylabel("dB")
        axes[row, 0].grid(True, alpha=0.3)

        im1 = axes[row, 1].imshow(spec_raw, aspect="auto", cmap="viridis")
        axes[row, 1].set_title(f"{name}: STFT raw (dB)")
        plt.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].imshow(spec_normed, aspect="auto", cmap="viridis")
        axes[row, 2].set_title(f"{name}: STFT normalised")
        plt.colorbar(im2, ax=axes[row, 2])

    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diag] Comparison plot saved to {fig_path}")


if __name__ == "__main__":
    main()