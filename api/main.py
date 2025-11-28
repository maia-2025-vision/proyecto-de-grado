import gc
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Never

from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse
from loguru import logger

import api.routes
from api.config import SETTINGS
from api.routes import router  # noqa: F401
from api.schemas.req_resp_types import PredictionError
from api.utils.model_utils import make_detector

DETECTOR = api.routes.DETECTOR


# Proper way to load a model on startup
# https://fastapi.tiangolo.com/advanced/events/#use-case
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Never]:
    # Load the ML model

    logger.info(f"MODEL_WEIGHTS_PATH={SETTINGS.model_weights_path}")
    logger.info(f"MODEL_CFG_PATH={SETTINGS.model_cfg_path}")

    logger.info(f"AWS_PROFILE={SETTINGS.aws_profile!r}")
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_is_defined = os.getenv("AWS_SECRET_ACCESS_KEY") is not None

    if SETTINGS.aws_profile is None and (aws_key_id is None or not aws_secret_is_defined):
        logger.info("No AWS_PROFILE or explicit keys: the IAM Task Role will be used if available.")

    global DETECTOR
    DETECTOR.detector, DETECTOR.model_metadata = make_detector(
        weights_path=SETTINGS.model_weights_path,
        cfg_path=SETTINGS.model_cfg_path,
    )

    yield  # type: ignore # this works but not sure what to do about type error...
    # Clean up the ML models and release the resources
    DETECTOR.detector = None  # type: ignore[assignment]  # this member will never be used from here on
    gc.collect()


app = FastAPI(title="Herd Detection API", lifespan=lifespan)
app.include_router(router, prefix="/api")


@app.exception_handler(PredictionError)
async def custom_exception_handler(request: requests.Request, exc: PredictionError) -> JSONResponse:
    req_body = await request.body()
    logger.error(f"request: {req_body!r}\nPredictionError: {exc}")
    return JSONResponse(
        status_code=exc.status,
        content={"url": exc.url, "error": str(exc), "traceback": traceback.format_exc()},
    )
