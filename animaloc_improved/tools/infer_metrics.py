import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

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
from torch import nn
from torch.utils.data import DataLoader

from animaloc_improved.tools.common import SPECIES_MAP


def load_trained_model(
    model_path: str | Path, num_classes: int = 7, down_ratio: int = 2
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


def predict_single_image(
    model: torch.nn.Module, image_path: str | Path, device: str = "cuda"
) -> dict:
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

            # print(f"Found {len(detections_copy)} predictions")

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

    pred_coords = predictions[["x", "y"]].values
    pred_labels = predictions["labels"].values
    pred_scores = predictions.get("scores", np.array([1.0] * len(predictions))).values

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
                        # added pred_label, gt_label for easier downstream processing
                        "pred_label": pred_label,
                        # gt_label: actually the same as pred_label due to condition
                        "gt_label": closest_gt_label,
                    }
                )
                gt_matched[closest_idx] = True
            else:
                # Wrong class prediction
                matches["false_positives"].append(
                    {
                        "coord": pred_coord,
                        "label": pred_label,  # added as the others have it
                        "pred_label": pred_label,
                        "gt_label": closest_gt_label,
                        "score": pred_score,
                        "distance": closest_distance,
                    }
                )
        else:
            # No close GT point or already matched
            matches["false_positives"].append(
                {
                    "coord": pred_coord,
                    "label": pred_label,
                    "pred_label": pred_label,
                    "score": pred_score,
                }
            )

    # Add unmatched GT points as false negatives
    for i, (gt_coord, gt_label) in enumerate(zip(gt_coords, gt_labels, strict=False)):
        if not gt_matched[i]:
            matches["false_negatives"].append(
                {"coord": gt_coord, "label": gt_label, "gt_label": gt_label}
            )

    return matches


def create_visualization(
    image_path: str | Path,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    matches: dict,
    colors: dict,
    species_map: dict,
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
            linewidth=1,
            label="Ground Truth" if coord is gt_coords[0] else "",
        )
        ax.add_patch(circle)
        ax.text(
            coord[0] + 10,
            coord[1] - 10,
            f"GT:{species_map[label]}",
            color=colors["ground_truth"],
            fontsize=6,
            weight="bold",
        )

    # Plot predictions
    if not predictions.empty:
        pred_coords = predictions[["x", "y"]].values
        pred_labels = predictions["labels"].values
        pred_scores = predictions.get("scores", [1.0] * len(predictions)).values

        for coord, label, score in zip(pred_coords, pred_labels, pred_scores, strict=False):
            circle = patches.Circle(
                coord,
                radius=3,
                color=colors["predictions"],
                fill=True,
                alpha=0.7,
                label="Predictions" if coord is pred_coords[0] else "",
            )
            ax.add_patch(circle)
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
            fp["coord"],
            radius=4,
            color=colors["false_positive"],
            fill=False,
            linewidth=1,
            linestyle="--",
        )
        ax.add_patch(circle)

    for fn in matches["false_negatives"]:
        circle = patches.Circle(
            fn["coord"], radius=4, color=colors["missed"], fill=False, linewidth=1, linestyle=":"
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
            markersize=10,
            label="Ground Truth",
            markeredgecolor=colors["ground_truth"],
            markeredgewidth=2,
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
            markerfacecolor="w",
            markersize=10,
            label="False Positives",
            markeredgecolor=colors["false_positive"],
            markeredgewidth=3,
            linestyle="--",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="w",
            markersize=10,
            label="Missed (FN)",
            markeredgecolor=colors["missed"],
            markeredgewidth=3,
            linestyle=":",
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

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Ground Truth Points: {len(ground_truth)}")
    print(f"Predicted Points: {len(predictions)}")
    print(f"True Positives: {n_tp}")
    print(f"False Positives: {n_fp}")
    print(f"False Negatives: {n_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    # Per-class breakdown
    print("\nPER-CLASS BREAKDOWN:")
    print("-" * 40)

    # Count GT by class
    gt_by_class = ground_truth["labels"].value_counts().sort_index()
    pred_by_class = (
        predictions["labels"].value_counts().sort_index() if not predictions.empty else pd.Series()
    )

    for class_id in sorted(set(list(gt_by_class.index) + list(pred_by_class.index))):
        species_name = species_map.get(class_id, f"Class_{class_id}")
        gt_count = gt_by_class.get(class_id, 0)
        pred_count = pred_by_class.get(class_id, 0)

        # Count TP for this class
        tp_class = sum(1 for tp in matches["true_positives"] if tp["label"] == class_id)

        print(f"{species_name}: GT={gt_count}, Pred={pred_count}, TP={tp_class}")

    print("=" * 60)


def predict_single_image_v2(*, model: nn.Module, image_path: Path, device: str) -> pd.DataFrame:
    """Run model detection on a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Make predictions
    prediction_results = predict_single_image(
        model=model,
        image_path=image_path,
        # lmds_kwargs=lmds_kwargs,
        device=device,
    )
    detections = prediction_results["detections"]

    if "x" in detections:  # i.e some actual detections
        return detections.sort_values("x")
    else:
        assert len(detections) == 1, f"{detections.head()=}"
        # ret = detections[["images"]].copy()
        # return empty dataframe with the right columns
        return pd.DataFrame({"images": [], "x": [], "y": [], "labels": [], "scores": []})


class PrecisionRecallEvaluator:
    """Can run detection on an image and then eval Precision-Recall with different strategies."""

    def __init__(
        self,
        match_strategy: str,
        match_tolerance: float,
        species_map: dict,
    ):
        self.match_strategy = match_strategy
        self.match_tolerance = match_tolerance
        self.species_map = species_map

    def evaluate_preds(
        self, *, ground_truth: pd.DataFrame, predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Call matching subroutine and then eval precision and recall."""
        # Match predictions to ground truth
        preds_binary = predictions.copy()
        # binary case animal vs. background
        preds_binary["labels"] = np.where(predictions["labels"] != 0, 1, 0)

        gt_binary = ground_truth.copy()
        # For test set gt labels is always != 0, but maybe in other case there is background
        gt_binary["labels"] = np.where(ground_truth["labels"] != 0, 1, 0)

        if self.match_strategy == "point2point":
            matches = match_predictions_to_gt(
                predictions, ground_truth, threshold=self.match_tolerance
            )
            matches_binary = match_predictions_to_gt(
                preds_binary, gt_binary, threshold=self.match_tolerance
            )
        else:
            raise NotImplementedError(
                f"Match strategy {self.match_strategy} is not implemented yet ."
            )

        evaluation = eval_precision_recall(
            matches=matches,
            ground_truth=ground_truth,
            predictions=predictions,
            species_map=self.species_map,
            is_binary=False,
        )

        binary_evaluation = eval_precision_recall(
            matches=matches_binary,
            ground_truth=gt_binary,
            predictions=preds_binary,
            species_map={},  # not used in binary case
            is_binary=True,
        )

        evaluation["matches"] = matches
        evaluation["matches_binary"] = matches_binary
        evaluation["binary"] = binary_evaluation

        return evaluation


class SpeciesStats(TypedDict):
    """Simple TP, FP, FN."""

    FN: int
    FP: int
    TP: int
    PP: int
    num_gt_annots: int
    num_pred_annots: int


def eval_precision_recall(
    matches: dict,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    species_map: dict[int, str],
    is_binary: bool,
) -> dict[str, int | float | dict[int, SpeciesStats]]:
    """Print detailed evaluation metrics."""
    n_tp = len(matches["true_positives"])
    n_fp = len(matches["false_positives"])
    n_fn = len(matches["false_negatives"])

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    result = {
        "num_gt_annots": len(ground_truth),
        "num_gt_positives": n_tp + n_fn,
        # Ground truth for test only has animals, never background
        # "num_gt_annots_animal": sum(ground_truth["labels"] != 0),
        # "num_gt_annots_background": sum(ground_truth["labels"] == 0),
        "num_preds": len(predictions),
        "TP": n_tp,
        "FP": n_fp,
        "PP": n_tp + n_fp,  # not necessarily equal to num_preds as there might be some preds
        "FN": n_fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }

    if not is_binary:
        # Per-class breakdown
        # Count GT by class

        gt_by_class = ground_truth["labels"].value_counts().sort_index()
        pred_by_class = (
            predictions["labels"].value_counts().sort_index()
            if not predictions.empty
            else pd.Series()
        )

        by_species: dict[int, dict[str, float]] = {}
        all_classes = set(list(gt_by_class.index) + list(pred_by_class.index))
        for class_id in sorted(all_classes):
            species_name = species_map.get(class_id, f"Class_{class_id}")
            gt_count = int(gt_by_class.get(class_id, 0))
            pred_count = int(pred_by_class.get(class_id, 0))

            # Count TP for this class
            tp_class = sum(1 for tp in matches["true_positives"] if tp["pred_label"] == class_id)
            fp_class = sum(1 for fp in matches["false_positives"] if fp["pred_label"] == class_id)
            fn_class = sum(1 for fn in matches["false_negatives"] if fn["gt_label"] == class_id)

            by_species[class_id]: SpeciesStats = {
                "species": species_name,
                "num_gt_annots": gt_count,
                "num_preds": pred_count,
                "TP": tp_class,
                "FP": fp_class,
                "PP": tp_class + fp_class,
                "FN": fn_class,
            }
            # print(f"{species_name}: GT={gt_count}, Pred={pred_count}, TP={tp_class}")

        result["by_species"] = by_species

    return result


def compute_by_image_results(
    evaluator: PrecisionRecallEvaluator, inference_results: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_image_results = []
    for inference in inference_results:
        image = inference["images"]
        gt_img_df = pd.DataFrame(**inference["ground_truth"])  # type: ignore
        predictions = pd.DataFrame(**inference["predictions"])  # type: ignore
        results_this_img = evaluator.evaluate_preds(ground_truth=gt_img_df, predictions=predictions)
        image_results = {
            "images": image,
            **results_this_img,
        }
        by_image_results.append(image_results)

    return by_image_results


@dataclass
class PrecisRecallResult:
    """Result of running Precision/Recall/F1 evaluation over many images."""

    label: 0
    species: str
    TP: int = 0
    FP: int = 0
    FN: int = 0
    cnt_updates = 0
    num_gt_annots: int = 0
    num_preds: int = 0
    precision: float = float("nan")
    recall: float = float("nan")
    f1_score: float = float("nan")

    def update(self, tfpn_dict: dict[str, int], num_gt_annots: int):
        """Tally contribution from one image."""
        self.cnt_updates += 1
        self.TP += tfpn_dict["TP"]
        self.FP += tfpn_dict["FP"]
        self.FN += tfpn_dict["FN"]
        self.num_preds += tfpn_dict["num_preds"]
        self.num_gt_annots += num_gt_annots

    def refresh_metrics(self):
        """After tallying all contributions, compute final value of precision/recall/f1-score."""
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN) if self.TP + self.FN > 0 else 0
        self.f1_score = (
            (2 * self.precision * self.recall / (self.precision + self.recall))
            if self.precision + self.recall > 0
            else 0
        )


def calc_precis_recall(
    by_image_results: list[object], species_map: dict[int, str] | None = None
) -> pd.DataFrame:
    pr_results: list[PrecisRecallResult] = []  # to be returned after converting it to pd.DataFrame
    species_map = species_map or SPECIES_MAP
    n_classes = len(species_map)

    for label in range(n_classes):
        if label == 0:  # binary case
            pr_res = PrecisRecallResult(label=0, species="binary")
            for result in by_image_results:
                bin_result = result["binary"]
                pr_res.update(bin_result, num_gt_annots=bin_result["num_gt_positives"])
            pr_res.refresh_metrics()

        else:
            pr_res = PrecisRecallResult(label=label, species=species_map[label])
            for result in by_image_results:
                by_species = result["by_species"]
                species_result = by_species.get(label, None)
                if species_result is not None:
                    pr_res.update(species_result, num_gt_annots=species_result["num_gt_annots"])
            pr_res.refresh_metrics()

        pr_results.append(pr_res)

    return pd.DataFrame([asdict(pr_res) for pr_res in pr_results])
