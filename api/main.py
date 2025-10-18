import gc
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import torchvision.transforms as transforms
from fastapi import FastAPI, requests
from fastapi.responses import JSONResponse
from loguru import logger

from api.config import SETTINGS
from api.model_utils import get_prediction_model
from api.req_resp_types import PredictionError
from api.routes import model_pack, router


# Proper way to load a model on startup
# https://fastapi.tiangolo.com/advanced/events/#use-case
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model

    logger.info(f"MODEL_PATH={SETTINGS.model_path}")

    logger.info(f"AWS_PROFILE={SETTINGS.aws_profile!r}")
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_is_defined = os.getenv("AWS_SECRET_ACCESS_KEY") is not None

    if SETTINGS.aws_profile is None and (aws_key_id is None or not aws_secret_is_defined):
        logger.error(
            "Need to provide at least AWS_PROFILE env var,"
            " or both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
        )
        raise RuntimeError("No AWS credentials!")

    pt_model = get_prediction_model(SETTINGS.model_path)
    pt_model.eval()

    model_pack.model = pt_model
    model_pack.pre_transform = transforms.ToTensor()
    # FIXME: set box_format for other models?
    model_pack.bbox_format = "xyxy"

    yield
    # Clean up the ML models and release the resources
    model_pack.model = None
    gc.collect()


app = FastAPI(title="Herd Detection API", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(PredictionError)
async def custom_exception_handler(request: requests.Request, exc: PredictionError):
    logger.error(f"request: {await request.body()}\nPredictionError: {exc}")
    return JSONResponse(
        status_code=exc.status,
        content={"url": exc.url, "error": str(exc), "traceback": traceback.format_exc()},
    )
