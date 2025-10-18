import pandas as pd
import os

def convert_csv_to_points(input_files, output_dir):
    """
    Convert CSV files with bounding box annotations to point annotations.

    Args:
        input_files (list): List of input CSV file paths.
        output_dir (str): Directory to save the converted CSV files.
    """
    for input_file in input_files:
        # Read the input CSV
        df = pd.read_csv(input_file)

        # Calculate the midpoint of the bounding boxes
        df['x'] = (df['x_min'] + df['x_max']) // 2
        df['y'] = (df['y_min'] + df['y_max']) // 2

        # Create the new DataFrame with the required format
        points_df = df[['images', 'x', 'y', 'labels']]

        # Generate the output file path
        base_name = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"points_{base_name}")

        # Save the new CSV
        points_df.to_csv(output_file, index=False)
        print(f"Converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_files = [
        "data/patches_512/gt/train_gt.csv",
        "data/patches_512/gt/val_gt.csv",
        "data/patches_512/gt/test_gt.csv"
    ]
    output_dir = "data/patches_512/gt"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the CSV files
    convert_csv_to_points(input_files, output_dir)