import argparse
import gc
import os
import sys
import time
import tracemalloc

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


def eval_image(
    model_path,
    csv_file,
    image_root,
    image_name,
    device,
    threshold,
    show_labels=True,
    figsize=None,
    dpi=300,
) -> None:
    species_map = {
        1: "Alcelaphinae",
        2: "Buffalo",
        3: "Kob",
        4: "Warthog",
        5: "Waterbuck",
        6: "Elephant",
    }

    colors = {
        "ground_truth": "white",
        "predictions": "lime",
        "correct": "blue",
        "missed": "orange",
        "false_positive": "red",
    }

    image_path = os.path.join(image_root, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = load_trained_model(model_path)
    ground_truth = get_single_image_data(csv_file, image_name)

    # Make predictions
    prediction_results = predict_single_image(model, image_path, device)
    predictions = prediction_results["detections"]

    # Match predictions to ground truth
    matches = match_predictions_to_gt(predictions, ground_truth, threshold)

    create_visualization(
        image_path,
        ground_truth,
        predictions,
        matches,
        colors,
        species_map,
        show_labels,
        figsize,
        dpi,
    )

    print_evaluation_results(matches, ground_truth, predictions, species_map)

    print("\nGround Truth Data:")
    print(ground_truth.to_string())

    print("\nPredictions:")
    print(predictions.to_string())


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

    gc.collect()
    tracemalloc.start()
    start_time = time.time()

    # GPU memory tracking
    initial_gpu_memory = 0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        print(f"Initial GPU memory: {initial_gpu_memory:.2f} MB")

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
            inference_start = time.time()
            # Run inference
            evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)
            inference_end = time.time()

            # Get detections
            detections = evaluator.detections
            detections = detections.dropna()
            detections_copy = detections.copy()

            # times
            end_time = time.time()
            total_time = end_time - start_time
            inference_time = inference_end - inference_start

            # Memory CPU
            current_cpu_memory, peak_cpu_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_cpu_memory_mb = peak_cpu_memory / 1024 / 1024
            current_cpu_memory_mb = current_cpu_memory / 1024 / 1024

            # Memory GPU
            gpu_memory_used = 0
            peak_gpu_memory = 0
            if device == "cuda" and torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                gpu_memory_used = abs(final_gpu_memory - initial_gpu_memory)

            print(f"\n{'=' * 50}")
            print("PERFORMANCE METRICS")
            print(f"{'=' * 50}")
            print(f"Total execution time: {total_time:.3f} seconds")
            print(f"Inference time: {inference_time:.3f} seconds")
            print(f"Current CPU Memory: {current_cpu_memory_mb:.2f} MB")
            print(f"Peak CPU Memory: {peak_cpu_memory_mb:.2f} MB")

            if device == "cuda" and torch.cuda.is_available():
                print(f"GPU Memory used: {gpu_memory_used:.2f} MB")
                print(f"Peak GPU Memory: {peak_gpu_memory:.2f} MB")
            else:
                print("Device: CPU (no GPU metrics)")

            return {"detections": detections_copy, "image_size": image.size}
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

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
    figsize: tuple = None,
    dpi: int = 100,
) -> None:
    """Create visualization showing GT, predictions, and matches."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    if figsize is None:
        width, height = image.size
        figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.imshow(image, interpolation="none")

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
                fontsize=3,
                weight="bold",
            )

    point_radius = 3 if show_labels else 7
    # Plot predictions
    if not predictions.empty and "x" in predictions.columns and "y" in predictions.columns:
        pred_coords = predictions[["x", "y"]].values
        pred_labels = predictions["labels"].values
        pred_scores = predictions.get("scores", [1.0] * len(predictions)).values

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
                    fontsize=3,
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
            fp["coord"], radius=point_radius, color=colors["false_positive"], fill=True
        )
        ax.add_patch(circle)

    for fn in matches["false_negatives"]:
        circle = patches.Circle(fn["coord"], radius=point_radius, color=colors["missed"], fill=True)
        ax.add_patch(circle)

    # Add legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linewidth=0,
            markeredgewidth=0.2,
            markerfacecolor=colors["ground_truth"],
            markersize=2,
            label="Ground Truth",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linewidth=0,
            markeredgewidth=0.2,
            markerfacecolor=colors["predictions"],
            markersize=2,
            label="Predictions",
        ),
        plt.Line2D([0], [0], color=colors["correct"], linewidth=0.5, label="Correct Matches"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            linewidth=0,
            markeredgewidth=0.2,
            markerfacecolor=colors["false_positive"],
            markersize=2,
            label="False Positives",
        ),
        plt.Line2D(
            [0],
            [0],
            linewidth=0,
            marker="o",
            color="k",
            markeredgewidth=0.2,
            markerfacecolor=colors["missed"],
            markersize=2,
            label="Missed (FN)",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1, 1),
        loc="upper left",
        fontsize=2,
        frameon=False,
    )
    ax.set_title(f"HerdNet Evaluation: {os.path.basename(image_path)}", fontsize=4, weight="bold")

    ax.set_xlim(0, image.size[0])
    ax.set_ylim(image.size[1], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    # Eliminar todos los márgenes
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate HerdNet model on single image")

    # Argumentos requeridos
    parser.add_argument("--model_path", required=True, help="Path to model file (.pth)")
    parser.add_argument("--csv_file", required=True, help="Path to CSV file with ground truth")
    parser.add_argument("--image_root", required=True, help="Root directory of images")
    parser.add_argument("--image_name", required=True, help="Name of image to evaluate")

    # Argumentos opcionales
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--threshold", type=float, default=15.0, help="Matching threshold (default: 15.0)"
    )
    parser.add_argument(
        "--show_labels", action="store_true", help="Show species labels in visualization"
    )
    parser.add_argument("--dpi", type=int, default=100, help="DPI for output image (default: 100)")

    args = parser.parse_args()

    # Validar que los archivos existen
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if not os.path.exists(args.image_root):
        print(f"Error: Image root directory not found: {args.image_root}")
        sys.exit(1)

    image_path = os.path.join(args.image_root, args.image_name)
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Ejecutar evaluación
    print(f"Evaluating image: {args.image_name}")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print("-" * 50)

    eval_image(
        model_path=args.model_path,
        csv_file=args.csv_file,
        image_root=args.image_root,
        image_name=args.image_name,
        device=args.device,
        threshold=args.threshold,
        show_labels=args.show_labels,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
