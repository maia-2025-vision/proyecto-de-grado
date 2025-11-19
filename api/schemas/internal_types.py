from dataclasses import dataclass

from api.detector import Detector
from api.schemas.shared_types import (
    BBoxFormat as _BBoxFormat,
)
from api.schemas.shared_types import (
    ModelMetadata as _ModelMetadata,
)

# Re-export shared aliases so other modules importing from this file keep working.
BBoxFormat = _BBoxFormat
ModelMetadata = _ModelMetadata


@dataclass
class DetectorHandle:
    """Just a container for a reference to a detector.

    This way we can have a single global object and overwrite that reference on lifespan.
    """

    detector: Detector
    model_metadata: ModelMetadata
