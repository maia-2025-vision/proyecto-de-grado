import os

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher
from animaloc.eval.metrics import PointsMetrics
from animaloc.models import HerdNet, LossWrapper, load_model
from PIL import Image
from torch.utils.data import DataLoader


def load_trained_model(
    model_path: str, num_classes: int = 7, down_ratio: int = 2
) -> torch.nn.Module:
    """Load the trained HerdNet model."""
    print(f"Loading model from: {model_path}")

    # Initialize model
    model = HerdNet(num_classes=num_classes, down_ratio=down_ratio)
    model = LossWrapper(model, losses=[])

    # Load trained weights
    model = load_model(model, model_path)
    model.eval()

    return model


def get_single_image_data(csv_file: str, image_name: str) -> pd.DataFrame:
    """Extract ground truth data for a specific image."""
    df = pd.read_csv(csv_file)
    image_data = df[df["images"] == image_name].copy()

    if image_data.empty:
        raise ValueError(f"No data found for image: {image_name}")

    print(f"Found {len(image_data)} ground truth annotations for {image_name}")
    return image_data


def predict_single_image(model: torch.nn.Module, image_path: str, device: str = "cuda") -> dict:
    """Make predictions on a single image using HerdNet."""
    print(f"Making predictions for: {os.path.basename(image_path)}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Create a dummy CSV entry for the image
    dummy_df = pd.DataFrame(
        {
            "images": [os.path.basename(image_path)],
            "x": [0],  # dummy coordinates
            "y": [0],
            "labels": [1],  # dummy label
        }
    )

    # Setup transforms (same as in configs)
    albu_transforms = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    end_transforms = [DownSample(down_ratio=2, anno_type="point")]

    # Create dataset and dataloader
    dataset = CSVDataset(
        csv_file=dummy_df,
        root_dir=os.path.dirname(image_path),
        albu_transforms=albu_transforms,
        end_transforms=end_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Setup evaluator with stitcher for inference
    device_obj = torch.device(device)

    metrics = PointsMetrics(radius=5, num_classes=7)

    stitcher = HerdNetStitcher(
        model=model,
        size=(512, 512),
        overlap=0,
        down_ratio=2,
        up=True,
        reduction="mean",
        device_name=device_obj,
    )

    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        device_name=device_obj,
        print_freq=1,
    )

    try:
        with torch.no_grad():
            # Run inference
            evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

            # Get detections
            detections = evaluator.detections
            detections = detections.dropna()

            detections_copy = detections.copy()

            return {"detections": detections_copy, "image_size": image.size}
    finally:
        # Clean up GPU memory
        if "evaluator" in locals():
            del evaluator
        if "stitcher" in locals():
            del stitcher
        if "metrics" in locals():
            del metrics
        if "dataloader" in locals():
            del dataloader
        if "dataset" in locals():
            del dataset

        # Force garbage collection
        import gc

        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def match_predictions_to_gt(
    predictions: pd.DataFrame, ground_truth: pd.DataFrame, threshold: float = 5.0
) -> dict:
    """Match predictions to ground truth based on distance threshold."""
    matches = {"true_positives": [], "false_positives": [], "false_negatives": []}

    # Check if predictions has individual detections (x, y columns)
    if "x" not in predictions.columns or "y" not in predictions.columns or predictions.empty:
        # No individual predictions found, all GT are false negatives
        gt_coords = ground_truth[["x", "y"]].values
        gt_labels = ground_truth["labels"].values

        for gt_coord, gt_label in zip(gt_coords, gt_labels, strict=False):
            matches["false_negatives"].append({"coord": gt_coord, "label": gt_label})

        return matches

    pred_coords = predictions[["x", "y"]].values
    pred_labels = predictions["labels"].values
    pred_scores = predictions.get("scores", [1.0] * len(predictions)).values

    gt_coords = ground_truth[["x", "y"]].values
    gt_labels = ground_truth["labels"].values

    # Track which GT points have been matched
    gt_matched = np.zeros(len(gt_coords), dtype=bool)

    # For each prediction, find closest GT point
    for _, (pred_coord, pred_label, pred_score) in enumerate(
        zip(pred_coords, pred_labels, pred_scores, strict=False)
    ):
        distances = np.sqrt(np.sum((gt_coords - pred_coord) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_gt_label = gt_labels[closest_idx]

        # Check if within threshold and labels match
        if closest_distance <= threshold and not gt_matched[closest_idx]:
            if pred_label == closest_gt_label:
                matches["true_positives"].append(
                    {
                        "pred_coord": pred_coord,
                        "gt_coord": gt_coords[closest_idx],
                        "label": pred_label,
                        "score": pred_score,
                        "distance": closest_distance,
                    }
                )
                gt_matched[closest_idx] = True
            else:
                # Wrong class prediction
                matches["false_positives"].append(
                    {
                        "coord": pred_coord,
                        "pred_label": pred_label,
                        "gt_label": closest_gt_label,
                        "score": pred_score,
                        "distance": closest_distance,
                    }
                )
        else:
            # No close GT point or already matched
            matches["false_positives"].append(
                {"coord": pred_coord, "label": pred_label, "score": pred_score}
            )

    # Add unmatched GT points as false negatives
    for i, (gt_coord, gt_label) in enumerate(zip(gt_coords, gt_labels, strict=False)):
        if not gt_matched[i]:
            matches["false_negatives"].append({"coord": gt_coord, "label": gt_label})

    return matches


def create_visualization(
    image_path: str,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    matches: dict,
    colors: dict,
    species_map: dict,
    show_labels: bool = True,
) -> None:
    """Create visualization showing GT, predictions, and matches."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)

    # Plot ground truth points
    gt_coords = ground_truth[["x", "y"]].values
    gt_labels = ground_truth["labels"].values

    for coord, label in zip(gt_coords, gt_labels, strict=False):
        circle = patches.Circle(
            coord,
            radius=5,
            color=colors["ground_truth"],
            fill=False,
            linewidth=1 if show_labels else 4,
        )
        ax.add_patch(circle)
        if show_labels:
            ax.text(
                coord[0] + 10,
                coord[1] - 10,
                f"GT:{species_map[label]}",
                color=colors["ground_truth"],
                fontsize=6,
                weight="bold",
            )

    # Plot predictions
    if not predictions.empty and "x" in predictions.columns and "y" in predictions.columns:
        pred_coords = predictions[["x", "y"]].values
        pred_labels = predictions["labels"].values
        pred_scores = predictions.get("scores", [1.0] * len(predictions)).values

        point_radius = 3 if show_labels else 7

        for coord, label, score in zip(pred_coords, pred_labels, pred_scores, strict=False):
            circle = patches.Circle(
                coord,
                radius=point_radius,
                color=colors["predictions"],
                fill=True,
                alpha=0.7,
            )
            ax.add_patch(circle)
            if show_labels:
                ax.text(
                    coord[0] + 10,
                    coord[1] + 10,
                    f"P:{species_map.get(label, 'Unknown')} ({score:.2f})",
                    color=colors["predictions"],
                    fontsize=6,
                    weight="bold",
                )

    # # Plot matches with connecting lines
    for tp in matches["true_positives"]:
        # Draw line connecting GT and prediction
        ax.plot(
            [tp["gt_coord"][0], tp["pred_coord"][0]],
            [tp["gt_coord"][1], tp["pred_coord"][1]],
            color=colors["correct"],
            linewidth=1,
            alpha=0.7,
        )

    # Highlight false positives and false negatives
    for fp in matches["false_positives"]:
        circle = patches.Circle(
            fp["coord"], radius=point_radius, color=colors["false_positive"], fill=True, alpha=0.7
        )
        ax.add_patch(circle)

    for fn in matches["false_negatives"]:
        circle = patches.Circle(
            fn["coord"], radius=point_radius, color=colors["missed"], fill=True, alpha=0.7
        )
        ax.add_patch(circle)

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["ground_truth"],
            markersize=8,
            label="Ground Truth",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["predictions"],
            markersize=8,
            label="Predictions",
        ),
        plt.Line2D([0], [0], color=colors["correct"], linewidth=2, label="Correct Matches"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["false_positive"],
            markersize=8,
            label="False Positives",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["missed"],
            markersize=8,
            label="Missed (FN)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    ax.set_title(f"HerdNet Evaluation: {os.path.basename(image_path)}", fontsize=14, weight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def print_evaluation_results(
    matches: dict, ground_truth: pd.DataFrame, predictions: pd.DataFrame, species_map: dict
) -> None:
    """Print detailed evaluation metrics."""
    n_tp = len(matches["true_positives"])
    n_fp = len(matches["false_positives"])
    n_fn = len(matches["false_negatives"])
    n_pred = len(predictions) if "x" in predictions.columns and "y" in predictions.columns else 0

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Ground Truth Points: {len(ground_truth)}")
    print(f"Predicted Points: {n_pred}")
    print(f"True Positives: {n_tp}")
    print(f"False Positives: {n_fp}")
    print(f"False Negatives: {n_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    # Count GT by class
    gt_by_class = ground_truth["labels"].value_counts().sort_index()

    # Only count predictions by class if individual predictions exist
    if not predictions.empty and "labels" in predictions.columns:
        pred_by_class = predictions["labels"].value_counts().sort_index()
    else:
        pred_by_class = pd.Series(dtype=int)

    # Per-class breakdown
    print("\nPER-CLASS BREAKDOWN:")
    print("-" * 40)

    for class_id in sorted(set(list(gt_by_class.index) + list(pred_by_class.index))):
        species_name = species_map.get(class_id, f"Class_{class_id}")
        gt_count = gt_by_class.get(class_id, 0)
        pred_count = pred_by_class.get(class_id, 0)

        # Count TP for this class
        tp_class = sum(1 for tp in matches["true_positives"] if tp["label"] == class_id)

        print(f"{species_name}: GT={gt_count}, Pred={pred_count}, TP={tp_class}")

    print("=" * 60)
