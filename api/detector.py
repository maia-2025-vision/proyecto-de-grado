from abc import ABC, abstractmethod
from typing import Literal

from PIL import Image

from api.model_utils import RawDetections


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
