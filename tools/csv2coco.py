import json
from pathlib import Path

import pandas as pd
from loguru import logger
from PIL import Image
from tqdm import tqdm
from typer import Option, Typer

cli = Typer(no_args_is_help=True)

EXPECTED_COLUMNS = {"images", "x_min", "y_min", "x_max", "y_max", "labels"}


@cli.command("convert")
def main(
    csv_path: Path,
    image_root_dir: Path,
    id2class_file: Path = Option(
        Path("data/groundtruth/id2class_name.json"),
        "--i2c",
        help="Path json file containing id to class mapping",
    ),
    output_path: Path = Option(..., "-o", "--out", help="Path to output json file"),
) -> None:
    # --- Load Data and Initialize COCO Structure ---
    df = pd.read_csv(csv_path)
    logger.info(f"Read df shape: {df.shape} from {csv_path!s}")

    # Assume your CSV columns are: ['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'class_name']
    # Adjust column names as needed for your specific CSV
    assert set(df.columns).issuperset(EXPECTED_COLUMNS), (
        f"{tuple(df.columns)} does not contain all expected columns"
    )

    coco_output: dict[str, list[dict[str, object]]] = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Process Categories ---
    unique_labels = sorted(df["labels"].unique())
    id2class = {int(i): name for i, name in json.loads(id2class_file.read_text()).items()}
    logger.info(f"id2class: {id2class}")
    # category_map = {name: i + 1 for i, name in enumerate(class_names)}
    coco_output["categories"] = [
        {"id": i, "name": class_name, "supercategory": "none"} for i, class_name in id2class.items()
    ]
    logger.info(f"Found {len(unique_labels)} unique labels.")
    for label in unique_labels:
        assert label in id2class.keys(), f"label '{label}' not found in id2class."

    # --- 2. Process Images and Annotations ---
    image_id = 0
    annotation_id = 0
    image_map = {}  # Map image_path to image_id

    # Group annotations by image for efficient processing
    grouped_df = df.groupby("images")

    logger.info("Processing images and annotations...")
    for img_rel_path, group in tqdm(grouped_df):
        full_path = image_root_dir / img_rel_path  # type: ignore[operator]

        # 2a. Get Image Dimensions
        try:
            with Image.open(full_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {full_path}: {e}")
            continue

        image_id += 1
        image_map[img_rel_path] = image_id

        # Add Image Entry
        coco_output["images"].append(
            {"id": image_id, "file_name": img_rel_path, "width": width, "height": height}
        )

        # 2b. Add Annotations for this Image
        for _, row in group.iterrows():
            # Convert from [x_min, y_min, x_max, y_max] to COCO [x, y, w, h]
            x_min, y_min = row["x_min"], row["y_min"]
            width = row["x_max"] - x_min
            height = row["y_max"] - y_min
            bbox_coco = [x_min, y_min, width, height]

            annotation_id += 1

            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(row["labels"]),
                    "bbox": bbox_coco,
                    "area": width * height,
                    "iscrowd": 0,  # Assuming bounding box annotations are not crowds
                }
            )

    logger.info(f"Total images: {len(coco_output['images'])}")
    logger.info(f"Total annotations: {len(coco_output['annotations'])}")

    # --- 3. Save COCO JSON ---
    with open(output_path, "w") as out_f:
        json.dump(coco_output, out_f, indent=4)

    logger.info(f"Successfully created COCO JSON at {output_path!s}")


if __name__ == "__main__":
    cli()
