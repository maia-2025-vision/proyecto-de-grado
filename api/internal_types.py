from dataclasses import dataclass
from typing import Literal

from api.detector import Detector

BBoxFormat = Literal["xyxy", "xywh"] | None


@dataclass
class DetectorHandle:
    """Just a container for a reference to a detector.

    This way we can have a single global object and overwrite that reference on lifespan.
    """

    detector: Detector
