from http import HTTPStatus
from typing import Annotated

from pydantic import BaseModel


class PredictionError(Exception):
    """Signals any error in the whole prediction process."""

    def __init__(self, url: str, status: HTTPStatus, error: str):
        super().__init__(error)
        self.url = url
        self.status = status


class PredictOneRequest(BaseModel):
    """Request for detection on one image."""

    s3_path: str


class PredictManyRequest(BaseModel):
    """Request for processing many images."""

    urls: list[str]


class Detections(BaseModel):
    """Boxes, and their scores in one image."""

    # parallel arrays:
    points: list[
        tuple[int, int]
    ]  # (x, y) points indicating each detected animal (or center of boxes if model generates boxes)
    scores: list[float]
    labels: list[int]  # labels correspond to different animal classes!
    boxes: list[list[float]] | None = None


class PredictionResult(BaseModel):
    """Result of predicting on one image."""

    url: str
    detections: Detections


class PredictManyResult(BaseModel):
    """Detection results for many images."""

    results: list[PredictionResult]


class ModelInfo(BaseModel):
    """Basic info about model used by endpoints."""

    path: str
    model_arch: str
    bbox_format: str | None


class AppInfoResponse(BaseModel):
    """App info response."""

    model_info: ModelInfo
    s3_bucket: str
