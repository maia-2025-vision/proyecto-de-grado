from abc import ABC, abstractmethod
from typing import Literal, NotRequired, TypedDict

import torch
from PIL import Image


class RawDetections(TypedDict):
    """Predictions coming directly from a model."""

    points: NotRequired[torch.Tensor]
    labels: torch.Tensor  # of ints...
    scores: torch.Tensor
    boxes: NotRequired[torch.Tensor]


class DetectionsDict(TypedDict):
    """Predictions coming directly from a model after converting to lists."""

    points: list[tuple[int, int]]
    labels: list[int]
    scores: list[float]
    boxes: NotRequired[list[list[float]]]
    total_detections: int


class Detector(ABC):
    """Abstract Base Class for detectors."""

    @abstractmethod
    def detect_one_image(self, img: Image.Image) -> RawDetections:
        """Get all detections on one image."""
        ...

    @abstractmethod
    def get_idx_2_species_dict(self) -> dict[int, str]:
        """Provide the idx-to-species-name mapping."""
        ...

    @abstractmethod
    def bbox_format(self) -> Literal["xywh", "xyxy"] | None:
        """Get bounding box format of detections."""
        ...
