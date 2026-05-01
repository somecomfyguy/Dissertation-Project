import numpy as np
import json
from modules.common.types import STFTParams, SAMPLE_RATE
from modules.common.spectrogram import compute_spectrogram, SpectrogramNormalizer
from modules.dataset_module.process_gateman import (
    scan_gateman_segments, read_gateman_chunk, read_gateman_clean_chunk,
)

stft_params = STFTParams()
spec_norm = SpectrogramNormalizer.load(
    "./Output/combined_spectrograms/normalization_stats.json")

print(f"Saved normaliser: mean={spec_norm.mean:.4f}, std={spec_norm.std:.4f}")

segs = scan_gateman_segments(
    "./modules/dataset_module/datasets/GatemanJamming", jsr_db=20.0)

# Check 20 segments per class
for label in ["clean", "jam_am", "jam_chirp"]:
    class_segs = [s for s in segs if s.label == label][:20]
    raw_means = []
    normed_means = []
    iq_rms_vals = []
    for seg in class_segs:
        if seg.dataset == "gateman_clean":
            parts = seg.scenario.split("|")
            iq = read_gateman_clean_chunk(
                seg.source_file, seg.start_sample, seg.num_samples)
        else:
            parts = seg.scenario.split("|")
            jammer_path = parts[1]
            jsr = float(parts[2].split("=")[1])
            iq = read_gateman_chunk(
                seg.source_file, jammer_path,
                seg.start_sample, seg.num_samples, jsr_db=jsr)
        
        iq_rms_vals.append(np.sqrt(np.mean(np.abs(iq)**2)))
        spec = compute_spectrogram(iq, SAMPLE_RATE, stft_params)
        raw_means.append(np.mean(spec))
        normed_means.append(np.mean(spec_norm.transform(spec)))
    
    print(f"\n{label}:")
    print(f"  IQ RMS:          {np.mean(iq_rms_vals):.1f}")
    print(f"  Spec raw mean:   {np.mean(raw_means):.2f} dB")
    print(f"  Spec normed mean:{np.mean(normed_means):.4f}")

print(f"\nReference: OAKBAT spec raw mean ≈ 43.34 dB, normed mean ≈ 0.757")