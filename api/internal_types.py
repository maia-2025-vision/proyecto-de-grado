from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from api.detector import Detector

BBoxFormat = Literal["xyxy", "xywh"] | None
Metadatum: TypeAlias = int | float | str | None | Path
ModelMetadata: TypeAlias = dict[str, Metadatum]


@dataclass
class DetectorHandle:
    """Just a container for a reference to a detector.

    This way we can have a single global object and overwrite that reference on lifespan.
    """

    detector: Detector
    model_metadata: ModelMetadata
