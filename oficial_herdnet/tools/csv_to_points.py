import argparse
import os

import pandas as pd
from loguru import logger


def convert_csv_to_points(input_file, output_file):
    """Convert CSV file with bounding box annotations to point annotations.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the converted CSV file.
    """
    # Read the input CSV
    logger.info(f"input file: {input_file}")
    df = pd.read_csv(input_file)
    in_columns = list(df.columns)
    logger.info(f"{len(df)} lines, columns= {', '.join(in_columns)}")

    # Calculate the midpoint of the bounding boxes
    df["x"] = (df["x_min"] + df["x_max"]) // 2
    df["y"] = (df["y_min"] + df["y_max"]) // 2

    # Create the new DataFrame with the required format
    points_df = df[["images", "x", "y", "labels"]]

    # Ensure the parent directory of the output file exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the new CSV
    points_df.to_csv(output_file, index=False)
    logger.info(
        f"{len(points_df)} lines (point annotations),"
        f" on {points_df['images'].nunique()} unique images"
    )
    columns = list(points_df.columns)
    logger.info(f"{len(columns)} output columns: {', '.join(columns)}")
    logger.info(f"output file: {output_file}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert CSV files with bounding box annotations to point annotations."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input CSV file containing bounding box annotations",
    )
    parser.add_argument("-o", "--output", required=True, help="Path to save the converted CSV file")

    # Parse arguments
    args = parser.parse_args()

    # Convert the CSV file
    convert_csv_to_points(args.input, args.output)
