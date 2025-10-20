import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class Settings:
    """Configuración principal del API."""

    host: str = "0.0.0.0"  # Acepta conexiones externas
    port: int = 8000  # Puerto por defecto
    reload: bool = True  # Recarga automática (solo para desarrollo)
    s3_bucket: str = "cow-detect-maia"
    model_path: Path = Path("./undefined")
    aws_profile: str | None = None


SETTINGS = Settings()

if "UVICORN_HOST" in os.environ:
    SETTINGS.host = os.environ["UVICORN_HOST"]

if "UVICORN_PORT" in os.environ:
    SETTINGS.port = int(os.environ["UVICORN_PORT"])

if "UVICORN_RELOAD" in os.environ:
    SETTINGS.reload = bool(os.environ["UVICORN_RELOAD"])

if "S3_BUCKET" in os.environ:
    SETTINGS.s3_bucket = os.getenv("S3_BUCKET", "cow-detect-maia")

if "MODEL_PATH" in os.environ:
    SETTINGS.model_path = Path(os.environ["MODEL_PATH"])

if "AWS_PROFILE" in os.environ:
    SETTINGS.aws_profile = os.environ["AWS_PROFILE"]

logger.info(f"SETTINGS: {SETTINGS}")
