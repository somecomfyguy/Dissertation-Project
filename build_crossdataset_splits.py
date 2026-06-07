"""
Step 1 — Build disjoint adapt/holdout splits for cross-dataset evaluation.

Partitions TEXBAT and GATEMAN into two time-disjoint portions per scenario:
    - adapt:   eligible to be mixed into training (first 50% by time)
    - holdout: used ONLY for evaluation, never seen in training (last 50%)

The split is per (scenario, label): within each scenario, each label's
segments are sorted by start_sample and cut at the 50% time mark. Because
windows are non-overlapping (20 ms, no overlap), a time cut is genuinely
disjoint — adapt and holdout never share a window.

IMPORTANT CAVEAT (recorded in the output for thesis honesty): GATEMAN has
one recording per (constellation, jammer) condition, so this is WINDOW-LEVEL
generalisation within a single recording, not generalisation to a new
recording. TEXBAT likewise splits within each scenario file.

Output: crossdataset_splits.json
    {
      "split_ratio": 0.5,
      "generalisation_level": "window-level (within-recording)",
      "texbat": { "<scenario>|<label>": <threshold_start_sample>, ... },
      "gateman": { "<scenario>|<label>": <threshold_start_sample>, ... }
    }

A segment is in ADAPT if start_sample <= threshold, else HOLDOUT.
"""
import json
from collections import defaultdict

from modules.dataset_module.process_texbat import scan_texbat_segments
from modules.dataset_module.process_gateman import scan_gateman_segments


TEXBAT_DIR   = "./modules/dataset_module/datasets/TexbatSpoofing"
GATEMAN_DIR  = "./modules/dataset_module/datasets/GatemanJamming"
GATEMAN_JSR  = 20.0
SPLIT_RATIO  = 0.5          # fraction assigned to adapt (first half by time)
OUTPUT_PATH  = "./crossdataset_splits.json"


def compute_thresholds(segments) -> dict:
    """
    For each (scenario, label) group, return the start_sample threshold at
    the SPLIT_RATIO mark of that group's time-ordered segments.

    Segments with start_sample <= threshold are adapt; the rest are holdout.
    """
    groups = defaultdict(list)
    for s in segments:
        key = f"{s.scenario}|{s.label}"
        groups[key].append(s.start_sample)

    thresholds = {}
    for key, starts in groups.items():
        starts_sorted = sorted(starts)
        cut_index = int(len(starts_sorted) * SPLIT_RATIO)
        # Clamp so both sides are non-empty when the group has >= 2 segments
        cut_index = max(1, min(cut_index, len(starts_sorted) - 1)) \
            if len(starts_sorted) >= 2 else len(starts_sorted)
        # Threshold is the last start_sample assigned to adapt.
        threshold = starts_sorted[cut_index - 1]
        thresholds[key] = int(threshold)
    return thresholds


def report_split(name, segments, thresholds):
    """Print adapt/holdout counts per label for inspection."""
    adapt = defaultdict(int)
    holdout = defaultdict(int)
    for s in segments:
        key = f"{s.scenario}|{s.label}"
        thr = thresholds[key]
        if s.start_sample <= thr:
            adapt[s.label] += 1
        else:
            holdout[s.label] += 1

    labels = sorted(set(list(adapt) + list(holdout)))
    print(f"\n[{name}] adapt / holdout counts per label:")
    for lbl in labels:
        a, h = adapt[lbl], holdout[lbl]
        total = a + h
        pct = (100.0 * a / total) if total else 0.0
        print(f"  {lbl:28s} adapt={a:7d}  holdout={h:7d}  "
              f"(adapt {pct:.1f}%)")
    print(f"  {'TOTAL':28s} adapt={sum(adapt.values()):7d}  "
          f"holdout={sum(holdout.values()):7d}")


def main():
    print("Scanning TEXBAT...")
    texbat_segs = scan_texbat_segments(TEXBAT_DIR)
    texbat_thr  = compute_thresholds(texbat_segs)
    report_split("TEXBAT", texbat_segs, texbat_thr)

    print("\nScanning GATEMAN...")
    gateman_segs = scan_gateman_segments(GATEMAN_DIR, jsr_db=GATEMAN_JSR)
    gateman_thr  = compute_thresholds(gateman_segs)
    report_split("GATEMAN", gateman_segs, gateman_thr)

    index = {
        "split_ratio": SPLIT_RATIO,
        "generalisation_level": "window-level (within-recording)",
        "rule": "segment is ADAPT if start_sample <= threshold else HOLDOUT",
        "texbat": texbat_thr,
        "gateman": gateman_thr,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nSplit index written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()