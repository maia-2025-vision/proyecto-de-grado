import io
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Literal, TypedDict

import requests
import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, HTTPException
from PIL import Image
from torch import nn

from api.config import SETTINGS
from api.model_utils import MockModel, verify_and_post_process_pred
from api.req_resp_types import (
    AppInfoResponse,
    Detections,
    ModelInfo,
    PredictionError,
    PredictionResult,
    PredictManyRequest,
    PredictManyResult,
    PredictOneRequest,
)
from api.s3_utils import (
    download_file_from_s3,
    get_predictions_from_s3_folder,
    list_flyover_folders,
    list_region_folders,
    upload_json_to_s3,
)


@dataclass
class ModelPackType:
    """Declare types for stuffed stored in model_pack global below."""

    model: nn.Module
    model_arch: str
    pre_transform: Callable[[Image.Image], torch.Tensor]
    bbox_format: Literal["xywh", "xyxy"] | None


# This gets properly initialized in api.main.lifespan
model_pack: ModelPackType = ModelPackType(
    model=MockModel(num_classes=0),
    model_arch="mock",
    pre_transform=transforms.ToTensor(),
    bbox_format="xyxy",
)

router = APIRouter()


def download_image_from_url(url: str):
    if url.startswith("s3://"):  # an s3 url that might not be public!
        file_bytes = download_file_from_s3(url)
    else:  # regular "public" url
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # lanza excepción si no es 200 OK
        file_bytes = response.content

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@router.get("/app-info")
async def get_app_info():
    return AppInfoResponse(
        model_info=ModelInfo(
            path=str(SETTINGS.model_path),
            model_arch=model_pack.model_arch,
            bbox_format=model_pack.bbox_format,
        ),
        s3_bucket=SETTINGS.s3_bucket,
    )


@router.post("/predict-many")
async def predict_many_endpoint(req: PredictManyRequest) -> PredictManyResult:
    """Realiza la predicción a partir de una lista de URLs de imágenes.

    Descarga cada imagen, la transforma en tensor, ejecuta el modelo de predicción
    y sube los resultados a S3. Devuelve una lista con los resultados o errores por imagen.
    """
    results = []

    for url in req.urls:
        try:
            result = predict_one(url)
        except PredictionError as err:
            result = PredictionError(url, status=HTTPStatus.INTERNAL_SERVER_ERROR, error=str(err))

        results.append(result)

    return PredictManyResult(results=results)


@router.post("/predict")
def predict_one_endpoint(req: PredictOneRequest) -> PredictionResult:
    return predict_one(url=req.s3_path)


def predict_one(url: str) -> PredictionResult:
    try:
        image = download_image_from_url(url)
    except Exception as exc:
        raise PredictionError(
            url=url,
            status=HTTPStatus.UNAUTHORIZED,
            error=f"No se pudo descargar o abrir la imagen: {str(exc)}",
        ) from exc

    image_tensor = model_pack.pre_transform(image).unsqueeze(0)  # batch size 1
    try:
        with torch.no_grad():
            model_outputs = model_pack.model(image_tensor)

        pred = model_outputs[0]  # just the first one since we only passed one image in the batch
        pred_obj = {k: v.tolist() for k, v in pred.items()}
        pred_obj2 = verify_and_post_process_pred(pred_obj, bbox_format=model_pack.bbox_format)
        pred_result = PredictionResult(url=url, detections=Detections.model_validate(pred_obj2))

    except Exception as exc:
        print(exc, traceback.format_exc())
        raise PredictionError(
            url=url,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            error=f"Error durante la predicción: {str(exc)}",
        ) from exc

    try:
        upload_json_to_s3(pred_result.model_dump(), url)
    except Exception as exc:
        raise PredictionError(
            url=url,
            status=HTTPStatus.UNAUTHORIZED,
            error=f"Error durante la subida a S3: {str(exc)}",
        ) from exc

    return pred_result


@router.get("/regions")
def list_regions() -> dict[str, list[str]]:
    """Lista las granjas que han generado datos de sobrevuelos."""
    try:
        return {"regions": list_region_folders()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al listar las regiones: {str(e)}"
        ) from e


@router.get("/flyovers/{region}")
def list_flyovers(region: str) -> dict[str, list[str]]:
    """Lista las carpetas de sobrevuelos disponibles para una región específica."""
    try:
        folders = list_flyover_folders(region=region)
        return {"flyovers": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar sobrevuelos: {str(e)}") from e


@router.get("/results/{region}/{flyover}")
def get_predictions_from_folder(region: str, flyover: str):
    """Obtiene las predicciones almacenadas en S3 para una region y sobrevuelo dadas."""
    try:
        results = get_predictions_from_s3_folder(region, flyover)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al reconstruir predicciones: {str(e)}"
        ) from e
