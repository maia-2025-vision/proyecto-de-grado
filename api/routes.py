import io
import traceback
from collections.abc import Callable
from http import HTTPStatus
from pprint import pformat
from typing import TypedDict, Literal

import requests
import torch
from fastapi import APIRouter, HTTPException
from PIL import Image
from torch import nn

from api.req_resp_types import (
    PredictionError,
    PredictionResult,
    PredictManyRequest,
    PredictManyResult,
    PredictOneRequest, Detections,
)
from api.s3_utils import (
    download_file_from_s3,
    get_predictions_from_s3_folder,
    list_region_folders,
    list_flyover_folders,
    upload_json_to_s3,
)


class ModelPackType(TypedDict):
    """Declare types for stuffed stored in model_pack global below."""

    model: nn.Module
    transform: Callable[[Image.Image], torch.Tensor]


# This gets initialized in api.main.lifespan
model_pack: ModelPackType = {}


router = APIRouter()


def download_image_from_url(url: str):
    if url.startswith("s3://"):  # an s3 url that might not be public!
        file_bytes = download_file_from_s3(url)
    else:  # regular "public" url
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # lanza excepción si no es 200 OK
        file_bytes = response.content

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@router.post("/predict-many")
async def predict_many_endpoint(req: PredictManyRequest):
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
    except Exception as e:
        raise PredictionError(
            url=url,
            status=HTTPStatus.UNAUTHORIZED,
            error=f"No se pudo descargar o abrir la imagen: {str(e)}",
        )

    transform, model = model_pack["transform"], model_pack["model"]

    image_tensor = transform(image).unsqueeze(0)  # batch size 1
    try:
        with torch.no_grad():
            model_outputs = model(image_tensor)

        pred = model_outputs[0]  # just the first one since we only passed one image in the batch
        pred_obj = {k: v.tolist() for k, v in pred.items()}
        # FIXME: set box_format for other models?
        pred_obj2 = verify_and_post_process_pred(pred_obj, box_format="PASCAL_VOC")
        pred_result = PredictionResult(
            url=url, detections=Detections.model_validate(pred_obj2)
        )

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
            url=url, status=HTTPStatus.UNAUTHORIZED, error=f"Error durante la subida a S3: {str(exc)}"
        ) from exc

    return pred_result


def verify_and_post_process_pred(
    pred: dict[str, list],
    box_format: Literal["COCO", "PASCAL_VOC"] | None
) -> dict[str, list]:
    """Make sure pred has a labels key AND (either boxes or points).

    If only boxes, compute box centers and add them.
    """
    assert "labels" in pred.keys(), f"{pred.keys()}"

    # logger.info(f"pred has keys: {pred.keys()}")
    if "boxes" not in pred:
        assert "points" in pred, f"Invalid pred: no bboxes and no point ins {pred.keys()=}"
    else:
        assert len(pred["boxes"]) == len(pred["labels"]), pformat(pred)

    if "points" not in pred:
        assert "boxes" in pred, f"{pred.keys()=}"

        if box_format == "COCO":
            # compute points from bboxes, assuming bbox in COCO format x_min, y_min, width, height
            points = [
                [bbox[0] + bbox[2] // 2,  # = x_min + width // 2 => x_center
                 bbox[1] + bbox[3] // 2]  # = y_min + height // 2 => x_center
                for bbox in pred["boxes"]
            ]
            pred["points"] = points
        elif box_format == "PASCAL_VOC":
            # compute points from bboxes, assuming bbox in COCO format x_min, y_min, x_max, y_max
            points = [
                [(bbox[0] + bbox[2]) // 2,  # = (x_min + x_max) // 2 => x_center
                 (bbox[1] + bbox[3]) // 2]  # = (y_min + y_max) // 2 => x_center
                for bbox in pred["boxes"]
            ]
            pred["points"] = points
        else:
            raise ValueError("box_format must be COCO or PASCAL_VOC when boxes are given")


    assert len(pred["points"]) == len(pred["labels"]), pformat(pred)

    return pred



@router.get("/regions")
def list_regions() -> dict[str, list[str]]:
    """Lista las granjas que han generado datos de sobrevuelos."""
    try:
        return {"regions": list_region_folders()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar las regiones: {str(e)}") from e


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
    """Obtiene las predicciones almacenadas en S3 para una granja y sobrevuelo dadas."""
    try:
        results = get_predictions_from_s3_folder(region, flyover)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al reconstruir predicciones: {str(e)}"
        ) from e
