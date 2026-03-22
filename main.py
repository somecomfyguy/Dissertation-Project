# from modules import dataset_module
from modules.dataset_module  import (
    scan_oakbat_segments, scan_swinney_segments,
    create_splits, compute_joint_normalization,
    save_dataset_streaming, STFTParams, SAMPLE_RATE,
)
from modules.nn_module import *


OAKBAT_DATASET_PATH = "OakbatSpoofing"
SWINNEY_DATASET_PATH = "SwinneySpoofing"


def prepare_datasets():
    stft_params = STFTParams()

    # Step 1: Scan (metadata only, no IQ loaded)
    oakbat_meta = scan_oakbat_segments("OakbatSpoofing", 
                                        constellations=["gps", "galileo"])
    swinney_meta = scan_swinney_segments("SwinneyJamming", "Training")
    swinney_meta += scan_swinney_segments("SwinneyJamming", "Testing")

    # Step 2: Combine and split
    combined = oakbat_meta + swinney_meta
    splits = create_splits(combined, balance_classes=True, 
                        max_per_class=1000)

    # Step 3: Joint normalization from training split only
    normalizer = compute_joint_normalization(
        [s for s in splits["train"] if s.dataset == "oakbat"],
        [s for s in splits["train"] if s.dataset == "swinney"],
        stft_params,
        output_path="./combined_spectrograms/normalization_stats.json",
    )

    # Step 4: Save everything with joint normalizer
    save_dataset_streaming(splits, "./combined_spectrograms",
                        SAMPLE_RATE, stft_params, normalizer)


def main():

    # Prepare the datasets for processing (should be done only once)
    prepare_datasets()

    # Train the neural networks on the spectrograms
    run_training(data_dir="./combined_spectrograms", output_dir="./Output")

if __name__ == "__main__":
    main()