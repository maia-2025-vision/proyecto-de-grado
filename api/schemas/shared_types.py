from pathlib import Path
from typing import Literal, TypeAlias

BBoxFormat = Literal["xyxy", "xywh"] | None
Metadatum: TypeAlias = int | float | str | None | Path
ModelMetadata: TypeAlias = dict[str, Metadatum]
