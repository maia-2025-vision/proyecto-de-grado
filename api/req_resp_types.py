from http import HTTPStatus
from typing import Annotated

from pydantic import BaseModel, Field


class PredictionError(Exception):
    """Signals any error in the whole prediction process."""

    def __init__(self, url: str, status: HTTPStatus, error: str):
        super().__init__(error)
        self.url = url
        self.status = status


class PredictOneRequest(BaseModel):
    """Request for detection on one image."""

    s3_path: Annotated[
        str, Field(description="The input image's uri on s3, in the form s3://bucket/path")
    ]
    counts_score_thresh: Annotated[
        float, Field(description="score threshold above which to include a detection in the counts")
    ] = 0.5


class PredictManyRequest(BaseModel):
    """Request for processing many images."""

    urls: Annotated[
        str, Field(description="The input images' uris on s3, in the form s3://bucket/path")
    ]
    counts_score_thresh: Annotated[
        float, Field(description="score threshold above which to include a detection in the counts")
    ] = 0.5


class Detections(BaseModel):
    """Boxes, and their scores in one image."""

    # parallel arrays:
    points: Annotated[
        list[tuple[int, int]],
        Field(
            description="(x, y) points indicating each detected animal "
            "(or center of boxes if model generates boxes)"
        ),
    ]
    scores: Annotated[
        list[float],
        Field(description="array parallel to points - indicates confidence of each detection"),
    ]
    labels: Annotated[
        list[int],
        Field(description="array parallel to points - containing indices of the detected classes"),
    ]
    boxes: Annotated[
        list[list[float]] | None,
        Field(
            description="array parallel to points - bounding boxes around animals "
            "(present for models that predict boxes)"
        ),
    ] = None


class ThresholdCounts(BaseModel):
    """Counts by species for detections having score above a certain threshold."""

    score_thresh: float
    counts: Annotated[
        dict[str, int],
        Field(description="map of species name to count of detections above the threshold"),
    ]


class PredictionResult(BaseModel):
    """Result of predicting on one image."""

    url: str
    detections: Detections
    counts_at_threshold: ThresholdCounts


class PredictManyResult(BaseModel):
    """Detection results for many images."""

    results: list[PredictionResult | PredictionError]


class ModelInfo(BaseModel):
    """Basic info about model used by endpoints."""

    path: str
    model_arch: str
    bbox_format: str | None


class AppInfoResponse(BaseModel):
    """App info response."""

    model_info: ModelInfo
    s3_bucket: str
