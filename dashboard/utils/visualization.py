import io
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
import streamlit as st
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont

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
def download_image(url: str) -> Image.Image:
    """Descarga una imagen desde una URL S3 o HTTP."""
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
    detections: dict,
    confidence_threshold: float,
    selected_labels: list[int],
    text_color: str = "#FFFFFF",
    line_width: int = 3,
) -> Image.Image:
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
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
