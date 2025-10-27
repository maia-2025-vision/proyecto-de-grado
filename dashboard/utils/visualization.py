"""Utilidades de visualización para dibujar detecciones en imágenes."""

import io
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
import streamlit as st
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

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


@st.cache_data
def download_image(url: str) -> Image.Image | None:
    """Descarga una imagen desde una URL S3 o HTTP y la devuelve como objeto PIL.

    Args:
        url: La URL de la imagen a descargar. Puede ser una URL `s3://` o `http(s)://`.

    Returns:
        Un objeto `PIL.Image.Image` si la descarga es exitosa, o `None` si falla.
    """
    if url.startswith("s3://"):
        try:
            s3_client = boto3.client("s3")
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
            st.error(f"Error inesperado al procesar imagen S3 ({url}): {e}")
            return None
    else:
        # Fallback for standard HTTP(S) URLs
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            st.error(f"No se pudo descargar la imagen desde {url}: {e}")
            return None


def draw_detections_on_image(
    image: Image.Image,
    detections: dict[str, Any],
    confidence_threshold: float,
    selected_labels: list[int],
    text_color: str = "#FFFFFF",
    line_width: int = 3,
) -> Image.Image:
    """Dibuja cajas delimitadoras (bounding boxes) y etiquetas sobre una imagen.

    Args:
        image: El objeto de imagen PIL sobre el cual dibujar.
        detections: Un diccionario con las detecciones, que debe contener
                    'boxes', 'labels' y 'scores'.
        confidence_threshold: El umbral de confianza mínimo para mostrar una detección.
        selected_labels: Una lista de IDs de etiquetas para filtrar qué especies mostrar.
        text_color: El color del texto de las etiquetas.
        line_width: El grosor de la línea de las cajas delimitadoras.

    Returns:
        Un nuevo objeto de imagen PIL con las detecciones dibujadas.
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    try:
        font: FreeTypeFont | ImageFont.ImageFont = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    scores = detections.get("scores", [])
    labels = detections.get("labels", [])
    boxes = detections.get("boxes", [])

    for i, score in enumerate(scores):
        label = labels[i]
        if score >= confidence_threshold and label in selected_labels:
            box_color = SPECIES_COLORS.get(label, "#FFFFFF")

            # Dibujar Bounding Box
            box = boxes[i]
            draw.rectangle(box, outline=box_color, width=line_width)

            # Preparar y dibujar etiqueta con fondo
            species_name = SPECIES_MAP.get(label, "Unknown")
            label_text = f"{species_name} ({score:.2f})"

            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Asegurarse de que el texto no se salga de la imagen
            text_x = box[0]
            text_y = box[1] - text_height - 5
            if text_y < 0:
                text_y = box[3] + 5

            draw.rectangle(
                [(text_x, text_y), (text_x + text_width + 4, text_y + text_height + 4)],
                fill=box_color,
            )
            draw.text((text_x + 2, text_y + 2), label_text, fill=text_color, font=font)

    return img_with_boxes


def draw_centroids_on_image(
    image: Image.Image,
    detections: dict[str, Any],
    confidence_threshold: float,
    selected_labels: list[int],
    point_size: int = 5,
) -> Image.Image:
    """Dibuja los centroides de las detecciones como puntos en una imagen.

    Args:
        image: El objeto de imagen PIL sobre el cual dibujar.
        detections: Un diccionario con las detecciones, que debe contener
                    'boxes', 'labels' y 'scores'.
        confidence_threshold: El umbral de confianza mínimo para mostrar una detección.
        selected_labels: Una lista de IDs de etiquetas para filtrar qué especies mostrar.
        point_size: El radio en píxeles de los puntos a dibujar.

    Returns:
        Un nuevo objeto de imagen PIL con los centroides dibujados.
    """
    img_with_points = image.copy()
    draw = ImageDraw.Draw(img_with_points)

    scores = detections.get("scores", [])
    labels = detections.get("labels", [])
    boxes = detections.get("boxes", [])

    for i, score in enumerate(scores):
        label = labels[i]
        if score >= confidence_threshold and label in selected_labels:
            box = boxes[i]
            xmin, ymin, xmax, ymax = box

            # Calcular el centroide
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            # Definir el color y el radio del punto
            point_color = SPECIES_COLORS.get(label, "#FFFFFF")
            radius = point_size

            # Dibujar el círculo (elipse con el mismo radio para x e y)
            draw.ellipse(
                (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
                fill=point_color,
                outline=point_color,
            )

    return img_with_points
