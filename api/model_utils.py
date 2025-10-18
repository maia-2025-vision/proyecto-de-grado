from pathlib import Path
from pprint import pformat
from typing import Literal

import numpy as np
import torch
import torchvision
from loguru import logger
from torch import nn
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
)

CLASS_LABEL_2_NAME = {
    0: "background",
    1: "PENDING-name-1",
    2: "PENDING-name-2",
    3: "PENDING-name-3",
    4: "PENDING-name-4",
    5: "PENDING-name-5",
    6: "PENDING-name-6",
}


def make_faster_rcnn_model(num_classes: int) -> FasterRCNN:  # type: ignore[no-any-unimported]
    """Get a faster-rcnn model with a box_predictor head for the given number of classes."""
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one that has the number of classes we need
    # (plus 1 for the background class)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class MockModel(nn.Module):
    """A model that just generates random predictions."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes  # total number of classes including background at index 0

    def __call__(self, image_batch: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Generate a list of predictions for a batch of images.

        It is assumed in the batch has the shape (batch_size, n_ch, height, width).
        Returns list of dicts of length batch_size
        """
        assert (
            image_batch.dim() == 4
        ), "expected shape of size 4 (batch_size, channels, height, width) but got {image.shape}"

        batch_size = image_batch.shape[0]

        results = []

        for _ in range(batch_size):
            height, width = image_batch.shape[-2:]

            n_detections = np.random.randint(low=20, high=50)

            xs = torch.randint(low=0, high=width, size=(n_detections, 1))
            ys = torch.randint(low=0, high=height, size=(n_detections, 1))

            labels = torch.randint(low=1, high=self.num_classes + 1, size=(n_detections,))
            scores = torch.rand(size=(n_detections,))  # uniform distribution on [0, 1)

            one_result = {"points": torch.hstack([xs, ys]), "labels": labels, "scores": scores}
            results.append(one_result)

            # just log:
            shapes = {k: v.shape for k, v in one_result.items()}
            logger.info(f"Mock model returning ret: {shapes}")

        return results


def determine_model_arch(weights_path: Path) -> Literal["faster-rcnn", "herdnet", "mock"]:
    weights_path_str = str(weights_path)

    for p in ["faster-rcnn", "herdnet", "mock"]:
        if p in weights_path_str:
            return p

    raise ValueError("Could not determine model architecture from model_path ")


def get_prediction_model(weights_path: Path) -> nn.Module:
    """Restore and return a prediction model from a weights file."""
    num_classes = len(CLASS_LABEL_2_NAME)

    model_arch = determine_model_arch(weights_path)

    if model_arch == "herdnet":
        raise NotImplementedError("Loading of Herdnet model not implemented yet...")
    elif model_arch == "faster-rcnn":
        model = make_faster_rcnn_model(num_classes=num_classes)
        logger.info(f"Loading weights from: {weights_path} onto faster-rcnn model")
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        assert isinstance(model, nn.Module)
        return model
    elif model_arch == "mock":
        model = MockModel(num_classes=num_classes)
        return model
    else:
        raise NotImplementedError(f"model_arch=`{model_arch}` not implemented yet")


def verify_and_post_process_pred(
    pred: dict[str, list], bbox_format: Literal["xyxy", "xywh"] | None
) -> dict[str, list]:
    """Make sure pred has a labels key AND (either boxes or points).

    If only boxes, compute box centers and add them.
    """
    assert "labels" in pred.keys(), f"{pred.keys()}"

    # logger.info(f"pred has keys: {pred.keys()}")
    if "boxes" not in pred:
        assert "points" in pred, f"Invalid pred: no bboxes and no point ins {pred.keys()=}"
    else:
        assert len(pred["boxes"]) == len(pred["labels"]), pformat(pred)

    if "points" not in pred:
        assert "boxes" in pred, f"{pred.keys()=}"

        if bbox_format == "xywh":
            # compute points from bboxes, assuming bbox in COCO format x_min, y_min, width, height
            points = [
                [
                    bbox[0] + bbox[2] // 2,  # = x_min + width // 2 => x_center
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

    return pred
