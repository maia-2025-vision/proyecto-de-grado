from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

app = typer.Typer()


@app.command(
    "from-pt-dets",
    help="mine hard negatives from a point detections file, "
    "such as one produced by herdnet inference",
)
def mine_hard_negs_points(
    gt_boxes_path: Path = typer.Option(..., "--gt_boxes", help="Path to ground truth boxes"),
    point_dets_path: Path = typer.Option(..., "--pt_dets", help="Path to detected points"),
    output_path: Path = typer.Option(..., "--out", help="Path where output file will be written"),
    iou_thresh: float = typer.Option(
        0.25, "--iou_thresh", help="IoU threshold for considering a detection a true positive"
    ),
) -> None:
    gt_boxes = pd.read_csv(gt_boxes_path)
    logger.info(f"Loaded {len(gt_boxes):,d} ground truth boxes from: {gt_boxes_path!s}")
    point_dets = pd.read_csv(point_dets_path)
    logger.info(f"Loaded {len(point_dets):,d} point detections from: {point_dets_path!s}")
    # avg_wh_by_label_df = compute_avg_wh_by_labels(gt_boxes)
    # box_dets = convert_detected_points_to_boxes(point_dets, avg_wh_by_label_df)
    # logger.info(f"Detections converted to boxes: {len():,d}: "
    #           f"[{', '.join(list(box_dets.columns))}]")

    gt_boxes_by_image = gt_boxes.groupby("images")

    def is_false_positive_point(point_det: pd.Series) -> pd.Series:
        boxes_in_image = gt_boxes_by_image.get_group(point_det["images"])

        # identify which boxes this point is contained in
        det_in_boxes = (
            (point_det["x"] >= boxes_in_image["x_min"])
            & (point_det["x"] <= boxes_in_image["x_max"])
            & (point_det["y"] >= boxes_in_image["y_min"])
            & (point_det["y"] <= boxes_in_image["y_max"])
        )

        return det_in_boxes.sum() == 0  # type: ignore[no-any-return]

    is_false_pos: pd.Series = point_dets.apply(is_false_positive_point, axis=1)
    logger.info(f"False positives detected: {is_false_pos.sum():,d}")

    gt_points = compute_gt_points(gt_boxes)
    # separate hard negs
    hard_negs = point_dets[is_false_pos].copy()
    hard_negs["labels"] = 0

    out = pd.concat([gt_points, hard_negs])
    lbl_counts = pd.DataFrame(out["labels"].value_counts()).sort_values("labels")
    logger.info(
        f"output = concat(gt_points, hard_negs), total rows: {len(out):,d} -- Label counts:\n"
        f"{lbl_counts.to_markdown()}\n"
    )

    logger.info(f"Writing detections to {output_path!s}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


# def is_false_positive_box(
#     box_det: pd.Series,
# ) -> pd.Series:
#     gt_boxes_in_image = gt_boxes_by_image.get_group(box_det["images"])
#     # compute iou for this box_det against gt boxes in the image in question
#     iou_boxes = compute_iou(box_det, gt_boxes_in_image)
#
#     # false positive if no boxes have iou above iou_thresh
#     return (iou_boxes > iou_thresh).sum() == 0  # type: ignore[no-any-return]


# def compute_iou(bbox: pd.Series, bboxes_df: pd.DataFrame) -> pd.Series:
#     """
#     Compute Intersection over Union (IoU) between a single bounding box
#     and multiple bounding boxes in a dataframe.
#
#     Parameters:
#     -----------
#     bbox : pd.Series
#         A single bounding box with fields: x_min, x_max, y_min, y_max
#     bboxes_df : pd.DataFrame
#         DataFrame of bounding boxes with columns: x_min, x_max, y_min, y_max
#
#     Returns:
#     --------
#     pd.Series
#         IoU scores for each bounding box in bboxes_df
#     """
#     # Extract coordinates of the single bbox
#     x1_min, x1_max = bbox['x_min'], bbox['x_max']
#     y1_min, y1_max = bbox['y_min'], bbox['y_max']
#
#     # Extract coordinates of all bboxes in the dataframe
#     x2_min = bboxes_df['x_min']
#     x2_max = bboxes_df['x_max']
#     y2_min = bboxes_df['y_min']
#     y2_max = bboxes_df['y_max']
#
#     # Compute intersection coordinates
#     inter_x_min = np.maximum(x1_min, x2_min)
#     inter_x_max = np.minimum(x1_max, x2_max)
#     inter_y_min = np.maximum(y1_min, y2_min)
#     inter_y_max = np.minimum(y1_max, y2_max)
#
#     # Compute intersection area
#     inter_width = np.maximum(0, inter_x_max - inter_x_min)
#     inter_height = np.maximum(0, inter_y_max - inter_y_min)
#     area_inter = inter_width * inter_height
#
#     # Compute areas of both bounding boxes
#     area1 = (x1_max - x1_min) * (y1_max - y1_min)
#     area2 = (x2_max - x2_min) * (y2_max - y2_min)
#
#     # Compute union area
#     area_union = area1 + area2 - area_inter
#
#     # Compute IoU
#     iou = area_inter / area_union
#
#     return iou


def compute_gt_points(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    """Compute ground truth points from a set of ground truth boxes.

    Implements same logic that csv_to_points script follows.
    """
    # Calculate the midpoint of the bounding boxes
    gt_boxes = gt_boxes.copy()
    gt_boxes["x"] = (gt_boxes["x_min"] + gt_boxes["x_max"]) // 2
    gt_boxes["y"] = (gt_boxes["y_min"] + gt_boxes["y_max"]) // 2

    # Create the new DataFrame with the required format
    points_df = gt_boxes[["images", "x", "y", "labels"]]

    return points_df


def compute_avg_wh_by_labels(gt_boxes: pd.DataFrame) -> pd.DataFrame:
    """Compute average width and height by label of a set of ground truth boxes."""
    gt_boxes["width"] = gt_boxes["x_max"] - gt_boxes["x_min"]
    gt_boxes["height"] = gt_boxes["y_max"] - gt_boxes["y_min"]
    avg_wh_df = gt_boxes.groupby("labels").agg(
        {
            "height": "mean",
            "width": "mean",
        }
    )

    return avg_wh_df


# def convert_detected_points_to_boxes(
#     pt_dets: pd.DataFrame, avg_wh_by_label_df: pd.DataFrame
# )-> pd.DataFrame:
#
#     box_dets = pt_dets.merge(avg_wh_by_label_df, left_on='labels', right_index=True, how='inner')
#     assert len(pt_dets) == len(box_dets)
#     box_dets['x_min'] = box_dets['x'] - box_dets['width'] / 2
#     box_dets['x_max'] = box_dets['x'] + box_dets['width'] / 2
#     box_dets['y_min'] = box_dets['y'] - box_dets['height'] / 2
#     box_dets['y_max'] = box_dets['y'] + box_dets['height'] / 2
#
#     return box_dets


@app.command("from-box-dets")
def mine_hard_negs_boxes(
    gt_boxes_path: Path = typer.Option(..., "--gt_boxes", help="Path to ground truth boxes"),
    box_dets_path: Path = typer.Option(..., "--pt_dets", help="Path to detected boxes"),
    output_path: Path = typer.Option(..., "--out", help="Path where output file will be written"),
) -> None:
    """Mine hard negatives from a box detections file.

    ...such as the one produced by faster-rcnn inference (?)
    """
    raise NotImplementedError()


if __name__ == "__main__":
    app()
