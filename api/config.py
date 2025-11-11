import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from loguru import logger
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings(
    "ignore",
    message=(
        "The '(repr|frozen)' attribute with value False was provided to the `Field\\(\\)` function"
    ),
    category=UnsupportedFieldAttributeWarning,
)


@dataclass
class Settings:
    """Configuración principal del API."""

    s3_bucket: str = "cow-detect-maia"
    model_weights_path: Path = Path("./undefined")
    model_cfg_path: Path = Path("./undefined")
    aws_profile: str | None = None


SETTINGS = Settings()

if "S3_BUCKET" in os.environ:
    SETTINGS.s3_bucket = os.environ["S3_BUCKET"]

if "MODEL_WEIGHTS_PATH" in os.environ:
    SETTINGS.model_weights_path = Path(os.environ["MODEL_WEIGHTS_PATH"])

if "MODEL_CFG_PATH" in os.environ:
    SETTINGS.model_cfg_path = Path(os.environ["MODEL_CFG_PATH"])

if "AWS_PROFILE" in os.environ:
    SETTINGS.aws_profile = os.environ["AWS_PROFILE"]

logger.info(f"SETTINGS:\n{pformat(asdict(SETTINGS))}")
