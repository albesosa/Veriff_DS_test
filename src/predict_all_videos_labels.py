#!/usr/bin/env python3
"""
This script processes all video result files in a specified directory and generates predicted labels.
"""

import os
import argparse
import pandas as pd
from src.video_processor import VideoSpoofProcessor


def predict_all_labels(output_data_dir, models_used):
    # Instantiate the processor with dummy parameters to use final_label_prediction
    processor = VideoSpoofProcessor(
        face_detector=None,
        spoof_detector=None,
        face_tracker=None,
        video_path=None,
        thr=None,
        show=None
    )

    # Collect results
    results = []
    for filename in os.listdir(output_data_dir):
        if f"{models_used}.csv" in filename:
            video_path = os.path.join(output_data_dir, filename)

            try:
                predicted_label = processor.final_label_prediction(video_path)

                # Extract video ID from filename
                basename = os.path.splitext(filename)[0]
                parts = basename.split('_')
                video_id = parts[1] if len(parts) > 1 else parts[0]

                results.append({
                    "video": video_id,
                    "predicted_label": predicted_label
                })

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save results to DataFrame and CSV
    df = pd.DataFrame(results)
    df["models"] = models_used
    output_file = os.path.join(output_data_dir, "predicted_labels.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predicted labels from result files.")
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default="output_data",
        help="Directory containing the result CSV files"
    )
    parser.add_argument(
        "--models_used",
        type=str,
        default="yolov3_deeppixbis_sort",
        help="Model name prefix to match result files"
    )

    args = parser.parse_args()
    predict_all_labels(args.output_data_dir, args.models_used)

# Example of how to run this script:
# python -m src.predict_all_videos_labels --output_data_dir output_data --models_used "yolov3_deeppixbis_sort"