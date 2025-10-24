import os
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from loguru import logger


@dataclass
class Settings:
    """Configuración principal del API."""

    host: str = "0.0.0.0"  # Acepta conexiones externas
    port: int = 8000  # Puerto por defecto
    reload: bool = True  # Recarga automática (solo para desarrollo)
    s3_bucket: str = "cow-detect-maia"
    model_weights_path: Path = Path("./undefined")
    model_cfg_path: Path = Path("./undefined")
    aws_profile: str | None = None


SETTINGS = Settings()

if "UVICORN_HOST" in os.environ:
    SETTINGS.host = os.environ["UVICORN_HOST"]

if "UVICORN_PORT" in os.environ:
    SETTINGS.port = int(os.environ["UVICORN_PORT"])

if "UVICORN_RELOAD" in os.environ:
    SETTINGS.reload = bool(os.environ["UVICORN_RELOAD"])

if "S3_BUCKET" in os.environ:
    SETTINGS.s3_bucket = os.environ["S3_BUCKET"]

if "MODEL_WEIGHTS_PATH" in os.environ:
    SETTINGS.model_weights_path = Path(os.environ["MODEL_WEIGHTS_PATH"])

if "MODEL_CFG_PATH" in os.environ:
    SETTINGS.model_cfg_path = Path(os.environ["MODEL_CFG_PATH"])

if "AWS_PROFILE" in os.environ:
    SETTINGS.aws_profile = os.environ["AWS_PROFILE"]

logger.info(f"SETTINGS:\n{pformat(SETTINGS)}")
