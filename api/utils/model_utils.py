from collections import Counter
from pathlib import Path
from pprint import pformat
from typing import Literal

import numpy as np
import torch
import torchvision
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from api.detector import DetectionsDict, Detector, RawDetections
from api.schemas.req_resp_types import ThresholdCounts
from api.schemas.shared_types import BBoxFormat, ModelMetadata
from api.utils.animaloc_utils import (
    faster_rcnn_detector_from_cfg_file,
    herdnet_detector_from_cfg_file,
)

DEFAULT_CLASS_LABEL_2_NAME = {
    0: "background",
    1: "Alcelaphinae",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}


class MockDetector(nn.Module, Detector):
    """A model that just generates random predictions."""

    def __init__(
        self,
        num_classes: int,
        idx2species: dict[int, str],
        bbox_format: Literal["xywh", "xyxy"] | None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes  # total number of classes including background at index 0
        self.idx2species = idx2species
        self.bbox_format_ = bbox_format
        self.to_tensor = torchvision.transforms.ToTensor()
        self.eval()

    def bbox_format(self) -> BBoxFormat:
        """Get bounding box format of generated detections."""
        return self.bbox_format_

    def detect_one_image(self, image: Image.Image) -> RawDetections:
        """Get detections on one image."""
        return self.detect_on_many([image])[0]

    def detect_on_many(self, image_batch: list[Image.Image]) -> list[RawDetections]:
        """Generate a list of predictions for a batch of images.

        It is assumed in the batch has the shape (batch_size, n_ch, height, width).
        Returns list of dicts of length batch_size
        """
        img_tensors = torch.vstack(
            [self.to_tensor(np.array(img.convert("RGB"))) for img in image_batch]
        )

        assert img_tensors.dim() == 4, (
            "expected shape of size 4 (batch_size, channels, height, width) but got {image.shape}"
        )

        batch_size = img_tensors.shape[0]
        results: list[RawDetections] = []

        for _ in range(batch_size):
            height, width = img_tensors.shape[-2:]

            n_detections = np.random.randint(low=20, high=50)

            xs = torch.randint(low=0, high=width, size=(n_detections, 1))
            ys = torch.randint(low=0, high=height, size=(n_detections, 1))

            labels = torch.randint(low=1, high=self.num_classes, size=(n_detections,))
            scores = torch.rand(size=(n_detections,))  # uniform distribution on [0, 1)

            one_result: RawDetections = {
                "points": torch.hstack([xs, ys]),
                "labels": labels,
                "scores": scores,
            }
            results.append(one_result)

            # just log:
            shapes = {k: v.shape for k, v in one_result.items()}  # type: ignore[attr-defined]
            logger.info(f"Mock model returning ret: {shapes}")

        return results

    def get_idx_2_species_dict(self) -> dict[int, str]:
        """Get mapping of label idx to species name."""
        return self.idx2species


def make_detector(weights_path: Path, cfg_path: Path) -> tuple[Detector, ModelMetadata]:
    """Restore and return a prediction model from a weights file."""
    num_classes = len(DEFAULT_CLASS_LABEL_2_NAME)
    cfg = OmegaConf.load(cfg_path)

    if cfg.model.name == "HerdNet":
        return herdnet_detector_from_cfg_file(
            model_pth_path=weights_path,
            cfg_file=cfg_path,
        )
    elif cfg.model.name == "FasterRCNNResNetFPN":
        return faster_rcnn_detector_from_cfg_file(
            model_pth_path=weights_path,
            cfg_file=cfg_path,
        )

    elif cfg.model.name == "mock":
        return MockDetector(
            num_classes=num_classes,
            idx2species=DEFAULT_CLASS_LABEL_2_NAME,
            bbox_format=None,
        ), dict(
            model_path=weights_path,
            model_arch="mock",
        )
    else:
        raise NotImplementedError(f"Loading of model.name=`{cfg.model.name}` not implemented yet")


def verify_and_post_process_pred(pred: DetectionsDict, bbox_format: BBoxFormat) -> DetectionsDict:
    """Make sure pred has a labels key AND (either boxes or points).

    If only boxes, compute box centers and add them.
    """
    assert "labels" in pred.keys(), f"{pred.keys()}"

    # logger.info(f"pred has keys: {pred.keys()}")
    if "boxes" not in pred:
        assert "points" in pred, f"Invalid pred: no bboxes and no points in {pred.keys()=}"
    else:
        assert len(pred["boxes"]) == len(pred["labels"]), pformat(pred)

    if "points" not in pred:
        assert "boxes" in pred, f"{pred.keys()=}"

        if bbox_format == "xywh":
            # compute points from bboxes, assuming bbox in COCO format x_min, y_min, width, height
            points = [
                [
                    # = x_min + width // 2 => x_center
                    bbox[0] + bbox[2] // 2,
                    bbox[1] + bbox[3] // 2,
                ]  # = y_min + height // 2 => x_center
                for bbox in pred["boxes"]
            ]
            pred["points"] = points
        elif bbox_format == "xyxy":
            # compute points from bboxes, assuming bbox in PASCAL_VOC format:
            # (x_min, y_min, x_max, y_max)
            points = [
                [
                    (bbox[0] + bbox[2]) // 2,  # = (x_min + x_max) // 2 => x_center
                    (bbox[1] + bbox[3]) // 2,
                ]  # = (y_min + y_max) // 2 => x_center
                for bbox in pred["boxes"]
            ]
            pred["points"] = points
        else:
            raise ValueError("box_format must be COCO or PASCAL_VOC when boxes are given")

    assert len(pred["points"]) == len(pred["labels"]), pformat(pred)
    pred["total_detections"] = len(pred["points"])

    return pred


def compute_counts_by_species(
    labels: list[int], scores: list[float], thresh: float, idx2species: dict[int, str]
) -> ThresholdCounts:
    assert len(labels) == len(scores)
    filtered_species = [
        idx2species[idx] for idx, scores in zip(labels, scores, strict=False) if scores > thresh
    ]
    counts = Counter(filtered_species)
    # Fill all missing slots with 0's for downstream tools to work more smoothly...
    for idx, species_name in idx2species.items():
        if idx == 0:  # background
            continue

        if species_name not in counts:
            counts[species_name] = 0

    return ThresholdCounts(
        score_thresh=thresh,
        counts=counts,
        total_count=sum(counts.values()),
    )
