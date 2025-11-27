"""Utilidades de visualización para dibujar detecciones en imágenes."""

import io
import os
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import requests
import streamlit as st
from botocore.exceptions import ClientError
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageDraw import ImageDraw as PILImageDraw
from PIL.ImageFont import FreeTypeFont

from api.schemas.req_resp_types import Detections
from dashboard.utils.api_client import ImageResults

Font: TypeAlias = FreeTypeFont | ImageFont.ImageFont

# Paleta de colores para las especies
SPECIES_COLORS = {
    1: "#FF0000",  # Rojo - Alcelaphinae
    2: "#00FF00",  # Verde - Buffalo
    3: "#0000FF",  # Azul - Kob
    4: "#FFFF00",  # Amarillo - Warthog
    5: "#800080",  # Morado - Waterbuck
    6: "#FFA500",  # Naranja - Elephant
    0: "#FFFFFF",  # Blanco - Unknown
}

SPECIES_MAP = {
    1: "Alcelaphinae",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
    0: "Unknown",
}


def ensure_size(thumb: Image.Image, w: int, h: int) -> Image.Image:
    """Agrega borde blanco a la derecha y abajo de una imagen para alcanzar el tamaño w x h.

    Args:
        thumb: Imagen a redimensionar
        w: Ancho objetivo
        h: Alto objetivo

    Returns:
        Nueva imagen de tamaño w x h con la imagen original en la esquina superior izquierda
    """
    current_w, current_h = thumb.size

    # Verificar que la imagen no sea más grande que el tamaño objetivo
    assert current_w <= w, f"El ancho actual ({current_w}) es mayor que el objetivo ({w})"
    assert current_h <= h, f"El alto actual ({current_h}) es mayor que el objetivo ({h})"

    # Si ya tiene el tamaño correcto, retornar la imagen original
    if current_w == w and current_h == h:
        return thumb

    # Crear una nueva imagen blanca del tamaño objetivo
    new_image = Image.new("RGB", (w, h), "white")

    # Pegar la imagen original en la esquina superior izquierda
    new_image.paste(thumb, (0, 0))

    return new_image


def create_thumbnail_with_marker(
    image: Image.Image, center: tuple[int, int], size: int = 200
) -> Image.Image:
    """Crea un thumbnail de tamaño fijo mostrando la imagen completa con un punto marcador.

    Args:
        image: Imagen original completa
        center: Coordenadas (x, y) del punto de detección
        size: Tamaño del thumbnail (cuadrado)

    Returns:
        Thumbnail de tamaño size x size con la imagen escalada y un punto marcador
    """
    from PIL import ImageDraw

    # Calcular el factor de escala para que la imagen quepa en el thumbnail
    width, height = image.size
    scale = min(size / width, size / height)

    # Calcular nuevas dimensiones manteniendo aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Escalar la imagen
    scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Crear imagen blanca del tamaño objetivo
    thumbnail = Image.new("RGB", (size, size), "white")

    # Centrar la imagen escalada
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2
    thumbnail.paste(scaled_image, (x_offset, y_offset))

    # Calcular posición del marcador en la imagen escalada
    marker_x = int(center[0] * scale) + x_offset
    marker_y = int(center[1] * scale) + y_offset

    # Dibujar punto marcador
    draw = ImageDraw.Draw(thumbnail)
    radius = max(3, int(size / 50))  # Radio proporcional al tamaño

    # Círculo rojo sin borde blanco
    draw.ellipse(
        [marker_x - radius, marker_y - radius, marker_x + radius, marker_y + radius],
        fill="red",
        outline=None,
        width=0,
    )

    return thumbnail


@st.cache_data
def download_image(url: str) -> Image.Image | None:
    """Descarga una imagen desde una URL S3 o HTTP y la devuelve como objeto PIL.

    Args:
        url: La URL de la imagen a descargar. Puede ser una URL `s3://` o `http(s)://`.

    Returns:
        Un objeto `PIL.Image.Image` si la descarga es exitosa, o `None` si falla.
    """
    logger.info(f"Descargando imagen {url}")
    if url.startswith("s3://"):
        try:
            s3_client = boto3.Session(profile_name=os.environ.get("AWS_PROFILE")).client("s3")
            parsed_url = urlparse(url)
            bucket_name = parsed_url.netloc
            object_key = parsed_url.path.lstrip("/")

            s3_response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            image_content = s3_response["Body"].read()

            return Image.open(io.BytesIO(image_content)).convert("RGB")
        except ClientError as e:
            st.error(f"Error al descargar desde S3 ({url}): {e}")
            return None
        except Exception as e:
            st.error(f"Error inesperado al descargar imagen S3 ({url}): {e}")
            return None
    else:
        # Fallback for standard HTTP(S) URLs
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            error_msg = f"No se pudo descargar la imagen desde {url}: {e}"
            st.error(error_msg)
            logger.error(error_msg)

            return None


def get_font() -> Font:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def draw_text_labels(
    draw: PILImageDraw,
    *,
    label: int,
    score: float,
    xmin: int,
    ymin: int,
    font: Font,
    text_color: str,
) -> None:
    species_name = SPECIES_MAP.get(label, f"ID_{label}")
    text = f"{species_name} ({score:.2f})"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = xmin
    text_y = ymin - text_height - 2

    if text_y < 0:
        text_y = 2

    box_color = SPECIES_COLORS.get(label, "#FFFFFF")
    draw.rectangle(
        [(text_x, text_y), (text_x + text_width + 4, text_y + text_height + 4)],
        fill=box_color,
    )
    draw.text((text_x, text_y), text, fill=text_color, font=font)


@dataclass
class AnnotParams:
    """Parametros para pintar cajar o puntos."""

    confidence_threshold: float
    selected_labels: list[int]
    add_text_boxes: bool = False
    line_width: int = 1
    point_size: int = 1
    text_color: str = "#FFFFFF"


def draw_boxes_on_image(
    image: Image.Image,
    detections: Detections,
    annot_params: AnnotParams,
) -> Image.Image:
    """Dibuja cajas delimitadoras (bounding boxes) y etiquetas sobre una imagen.

    Args:
        image: El objeto de imagen PIL sobre el cual dibujar.
        detections: Un diccionario con las detecciones, que debe contener
                    'boxes', 'labels' y 'scores'.
        annot_params: Parameters for drawing bounding boxes.

    Returns:
        Un nuevo objeto de imagen PIL con las detecciones dibujadas.
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    scores_raw = detections.scores or []
    labels_raw = detections.labels or []
    boxes_raw = detections.boxes or []

    scores = list(scores_raw)
    labels = list(labels_raw)
    boxes = list(boxes_raw)

    geometry = boxes if boxes else []
    max_len = min(len(scores), len(labels), len(geometry)) if geometry else 0

    font = get_font()
    for i in range(max_len):
        score = scores[i]
        label = labels[i]
        box = boxes[i]

        if score is None or box is None:
            continue

        draw_label = not annot_params.selected_labels or label in annot_params.selected_labels
        if score >= annot_params.confidence_threshold and draw_label:
            box_color = SPECIES_COLORS.get(label, "#FFFFFF")
            xmin, ymin, xmax, ymax = box

            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)], outline=box_color, width=annot_params.line_width
            )

            if annot_params.add_text_boxes:
                draw_text_labels(
                    draw,
                    label=label,
                    score=score,
                    xmin=int(xmin),
                    ymin=int(ymin),
                    font=font,
                    text_color=annot_params.text_color,
                )
    return img_with_boxes


def draw_points_on_image(
    image: Image.Image,
    detections: Detections,
    annot_params: AnnotParams,
) -> Image.Image:
    """Dibuja los puntos de las detecciones como puntos en una imagen.

    Args:
        image: El objeto de imagen PIL sobre el cual dibujar.
        detections: Un diccionario con las detecciones, que debe contener
                    'points', 'labels' y 'scores'.
        annot_params: Parameters for drawing points.

    Returns:
        Un nuevo objeto de imagen PIL con los centroides dibujados.
    """
    img_with_points = image.copy()
    draw = ImageDraw.Draw(img_with_points)

    scores_raw = detections.scores or []
    labels_raw = detections.labels or []
    points_raw = detections.points or []

    scores = list(scores_raw)
    labels = list(labels_raw)
    points = list(points_raw)

    geometry_len = len(points)  #  if use_points else len(boxes)

    max_len = min(len(scores), len(labels), geometry_len)

    font = get_font()

    for i in range(max_len):
        score = scores[i]
        label = labels[i]
        coords = points[i]

        if score is None or coords is None:
            continue

        is_label_selected = (
            not annot_params.selected_labels or label in annot_params.selected_labels
        )
        if score >= annot_params.confidence_threshold and is_label_selected:
            center_x, center_y = coords

            point_color = SPECIES_COLORS.get(label, "#FFFFFF")
            radius = annot_params.point_size

            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                # fill=point_color,
                fill=None,  # empty elipse
                outline=point_color,
                width=annot_params.line_width,
            )

            # draw cross-hairs:
            draw.line(
                [(center_x, center_y - radius), (center_x, center_y + radius)],
                fill=point_color,
                width=annot_params.line_width,
            )

            draw.line(
                [(center_x - radius, center_y), (center_x + radius, center_y)],
                fill=point_color,
                width=annot_params.line_width,
            )

            if annot_params.add_text_boxes:
                draw_text_labels(
                    draw,
                    label=label,
                    score=score,
                    xmin=center_x + radius + 2,
                    ymin=center_y + radius + 2,
                    font=font,
                    text_color=annot_params.text_color,
                )

    return img_with_points


def render_summary_tables(img_results: ImageResults, annot_params: AnnotParams) -> None:
    dets_df = pd.DataFrame(
        {"labels": img_results.detections.labels, "score": img_results.detections.scores}
    )

    dets_df = dets_df[dets_df.score >= annot_params.confidence_threshold]

    dets_df["Especie"] = dets_df["labels"].map(lambda lbl: SPECIES_MAP.get(lbl, "Unknown"))
    dets_df["Confianza [%]"] = np.round(dets_df["score"] * 100.0, 1)

    conteo_por_especies = dets_df.groupby("Especie").agg({"labels": "count"}).reset_index()

    st.subheader("Conteos por Especie")
    st.dataframe(conteo_por_especies, hide_index=True)

    st.subheader("Detecciones individuales")
    dets_df2 = dets_df.sort_values("score", ascending=False)[["Especie", "Confianza [%]"]]
    st.dataframe(dets_df2)
