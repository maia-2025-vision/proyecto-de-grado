from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import nn


@dataclass
class ModelPackType:
    """Declare types for stuffed stored in model_pack global below."""

    model: nn.Module
    model_arch: str
    model_path: Path
    pre_transform: Callable[[Image.Image], torch.Tensor]
    bbox_format: Literal["xywh", "xyxy"] | None
    idx2species: dict[int, str]  # Map of label int index to species name
