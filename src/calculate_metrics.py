"""This script calculates precision, recall, and F1 score based on actual and predicted labels."""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse


def evaluate_predictions(actual_labels_path, predicted_labels_path, output_path):
    # Load data
    actual_labels_df = pd.read_csv(actual_labels_path, delimiter="\t")
    predicted_labels_df = pd.read_csv(predicted_labels_path)

    # Ensure proper column names
    actual_labels_df.columns = ["video", "actual_label"]
    predicted_labels_df.columns = ["video", "predicted_label", "models_names"]

    # Merge both DataFrames
    merged_labels_df = actual_labels_df.merge(predicted_labels_df, on="video", how="left")
    merged_labels_df.to_csv(output_path, index=False)
    print(merged_labels_df)

    # Calculate metrics
    precision = precision_score(merged_labels_df['actual_label'], merged_labels_df['predicted_label'])
    recall = recall_score(merged_labels_df['actual_label'], merged_labels_df['predicted_label'])
    f1 = f1_score(merged_labels_df['actual_label'], merged_labels_df['predicted_label'])

    # Print the results
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate prediction performance from actual and predicted labels.")
    parser.add_argument(
        "--actual",
        type=str,
        default="data/labels.txt",
        help="Path to the actual labels TXT file (default: ../data/labels.txt)"
    )
    parser.add_argument(
        "--predicted",
        type=str,
        default="output_data/predicted_labels_submitetd.csv",
        help="Path to the predicted labels CSV file (default: ../output_data/predicted_labels_submitetd.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_data/merged_labels_submitted.csv",
        help="Path to save merged labels CSV (default: ../output_data/merged_labels_submitted.csv)"
    )

    args = parser.parse_args()

    evaluate_predictions(args.actual, args.predicted, args.output)

# Example of how to run this script:
# python -m src.calculate_metrics