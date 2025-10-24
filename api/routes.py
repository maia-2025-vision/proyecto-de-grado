import io
import traceback
from collections import Counter
from http import HTTPStatus

import pydantic
import requests
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from PIL import Image

from api.config import SETTINGS
from api.detector import DetectionsDict
from api.internal_types import DetectorHandle
from api.model_utils import (
    MockDetector,
    compute_counts_by_species,
    verify_and_post_process_pred,
)
from api.req_resp_types import (
    AppInfoResponse,
    CollectedCountsFlyover,
    CollectedCountsRegion,
    CountsRow,
    Detections,
    FlyoverCountsRow,
    ModelInfo,
    PredictionApiError,
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

# This structure gets properly initialized in api.main.lifespan
DETECTOR = DetectorHandle(
    detector=MockDetector(idx2species={}, num_classes=0, bbox_format=None), model_metadata={}
)

router = APIRouter()


def download_image_from_url(url: str) -> Image.Image:
    if url.startswith("s3://"):  # an s3 url that might not be public!
        file_bytes = download_file_from_s3(url)
    else:  # regular "public" url
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # lanza excepción si no es 200 OK
        file_bytes = response.content

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def try_download_image(url: str) -> Image.Image:
    try:
        return download_image_from_url(url)
    except Exception as exc:
        raise PredictionError(
            url=url,
            status=HTTPStatus.UNAUTHORIZED,
            error=f"No se pudo descargar o abrir la imagen: {str(exc)}",
        ) from exc


@router.get("/app-info")
async def get_app_info() -> AppInfoResponse:
    return AppInfoResponse(
        model_info=ModelInfo(
            weights_path=str(SETTINGS.model_weights_path),
            cfg_path=str(SETTINGS.model_cfg_path),
            model_metadata=DETECTOR.model_metadata,
        ),
        s3_bucket=SETTINGS.s3_bucket,
    )


@router.post("/predict-many")
async def predict_many_endpoint(req: PredictManyRequest) -> PredictManyResult:
    """Realiza la predicción a partir de una lista de URLs de imágenes.

    Descarga cada imagen, la transforma en tensor, ejecuta el modelo de predicción
    y sube los resultados a S3. Devuelve una lista con los resultados o errores por imagen.
    """
    results: list[PredictionResult | PredictionApiError] = []

    result: PredictionResult | PredictionApiError
    for url in req.urls:
        try:
            image = try_download_image(url)
            result = predict_one(image=image, url=url, counts_score_thresh=req.counts_score_thresh)
        except PredictionError as err:
            result = PredictionApiError.from_prediction_error(err)

        results.append(result)

    return PredictManyResult(results=results)


@router.post("/predict", description="Run detection on an image already uploaded to s3")
def predict_one_endpoint(req: PredictOneRequest) -> PredictionResult:
    url = req.s3_path
    image = try_download_image(url=url)
    return predict_one(image=image, url=url, counts_score_thresh=req.counts_score_thresh)


@router.post(
    path="/predict-on-upload",
    description="Run detection on an image that is uploaded directly to the endpoint"
    "(used for more direct testing)",
)
async def predict_on_uploaded_image(
    counts_score_thresh: float = 0.7,
    file: UploadFile = File("Uploaded image file"),
) -> PredictionResult:
    logger.info("About to read uploaded file")
    file_bytes: bytes = await file.read()

    image = Image.open(io.BytesIO(file_bytes))
    return predict_one(
        image=image,
        url="__undefined__",
        counts_score_thresh=counts_score_thresh,
        upload_result_to_s3=False,
    )


def predict_one(
    image: Image.Image, url: str, *, counts_score_thresh: float, upload_result_to_s3: bool = True
) -> PredictionResult:
    detector = DETECTOR.detector
    try:
        pred = detector.detect_one_image(image)
        pred_obj: DetectionsDict = {k: v.tolist() for k, v in pred.items()}  # type: ignore
        pred_obj2 = verify_and_post_process_pred(pred_obj, bbox_format=detector.bbox_format())

        counts_at_thresh = compute_counts_by_species(
            labels=pred_obj2["labels"],
            scores=pred_obj2["scores"],
            thresh=counts_score_thresh,
            idx2species=detector.get_idx_2_species_dict(),
        )

        pred_result = PredictionResult(
            url=url,
            detections=Detections.model_validate(pred_obj2),
            counts_at_threshold=counts_at_thresh,
        )

    except Exception as exc:
        print(exc, traceback.format_exc())
        raise PredictionError(
            url=url,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            error=f"Error durante la predicción: {str(exc)}",
        ) from exc

    if not upload_result_to_s3:
        return pred_result

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
def get_predictions_from_folder(region: str, flyover: str) -> dict[str, list[dict[str, object]]]:
    """Obtiene las predicciones almacenadas en S3 para una region y sobrevuelo dadas."""
    try:
        results = get_predictions_from_s3_folder(region, flyover)
        return {"results": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al reconstruir predicciones: {str(e)}"
        ) from e


@router.get("/counts/{region}/{flyover}")
def collect_counts_for_flyover(region: str, flyover: str) -> CollectedCountsFlyover:
    """Reune los diccionarios de conteo por especie de todas las imagenes de un sobrevuelo dada."""
    results = get_predictions_from_s3_folder(region, flyover)

    rows: list[CountsRow] = []
    total_counts: Counter[str] = Counter()

    for result in results:
        try:
            pred_result = PredictionResult.model_validate(result)
        except pydantic.ValidationError:
            # logger.warning(f"Validation error: result='{str(result)[:100]}...'")
            continue

        row = CountsRow(
            url=pred_result.url,
            counts_at_threshold=pred_result.counts_at_threshold,
        )
        total_counts += Counter(pred_result.counts_at_threshold.counts)
        rows.append(row)

    return CollectedCountsFlyover(
        region=region,
        flyover=flyover,
        total_counts=total_counts,
        rows=rows,
    )


@router.get("/counts/{region}")
def collect_counts_for_region(region: str) -> CollectedCountsRegion:
    """Reune los diccionarios de conteos por especie de todas las imagenes de una region.

    (sobre todos los sobrevuelos)
    """
    flyovers = list_flyover_folders(region=region)
    logger.info(f"Flyovers for region={region}, flyovers={flyovers}")

    rows: list[FlyoverCountsRow] = []

    totals_by_flyover: dict[str, Counter[str]] = {}
    grand_totals: Counter[str] = Counter()

    for flyover in flyovers:
        results = get_predictions_from_s3_folder(region, flyover)
        totals_by_flyover[flyover] = Counter()

        for result in results:
            try:
                pred_result = PredictionResult.model_validate(result)
            except pydantic.ValidationError:
                # logger.warning(f"Validation error: result='{str(result)[:100]}...'")
                continue

            row = FlyoverCountsRow(
                flyover=flyover,
                url=pred_result.url,
                counts_at_threshold=pred_result.counts_at_threshold,
            )
            totals_by_flyover[flyover] += Counter(pred_result.counts_at_threshold.counts)
            rows.append(row)
        # End of loop over results of flyover
        grand_totals += totals_by_flyover[flyover]

    return CollectedCountsRegion(
        grand_totals=grand_totals,
        totals_by_flyover=totals_by_flyover,
        region=region,
        rows=rows,
    )
