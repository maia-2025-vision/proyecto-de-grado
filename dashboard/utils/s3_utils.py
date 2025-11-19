"""Utilidades para la interacción con S3."""

import json
from datetime import UTC, date, datetime
from uuid import uuid4

import boto3
import streamlit as st
from botocore.exceptions import NoCredentialsError
from mypy_boto3_s3.client import S3Client

# El nombre del bucket debería coincidir con el configurado en la API
S3_BUCKET_NAME = "cow-detect-maia"


@st.cache_resource
def get_s3_client() -> S3Client | None:
    """Crea y devuelve un cliente de S3 cacheado."""
    try:
        return boto3.client("s3")
    except NoCredentialsError:
        st.error("Error de credenciales de AWS. Asegúrate de tenerlas configuradas.")
        return None


def upload_files_to_s3(
    files: list[st.runtime.uploaded_file_manager.UploadedFile],
    region: str,
    flyover_date: date,
) -> list[str] | None:
    """Sube una lista de archivos a una carpeta estructurada en S3 y devuelve sus URLs.

    La estructura es /<region>/<flyover_timestamp>/<filename>.
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None

    flyover_ts = flyover_date.isoformat()
    s3_urls = []

    progress_bar = st.progress(0.0, text="Iniciando subida a S3...")

    for i, file in enumerate(files):
        try:
            # La ruta en S3 debe coincidir con la que la API espera para organizar los datos
            object_name = f"{region}/{flyover_ts}/{file.name}"

            file.seek(0)  # Asegurarse de que el puntero del archivo está al inicio
            s3_client.upload_fileobj(file, S3_BUCKET_NAME, object_name)

            s3_url = f"s3://{S3_BUCKET_NAME}/{object_name}"
            s3_urls.append(s3_url)

            progress = (i + 1) / len(files)
            progress_bar.progress(progress, text=f"Subiendo '{file.name}' a S3...")

        except Exception as e:
            st.error(f"Error al subir '{file.name}': {e}")
            progress_bar.empty()
            return None

    progress_bar.empty()
    return s3_urls


def upload_feedback_payload(
    *,
    region: str,
    flyover: str,
    records: list[dict[str, object]],
) -> str | None:
    """Sube un archivo JSON con las reclasificaciones hechas desde el dashboard."""
    if not records:
        st.warning("No hay feedback para subir.")
        return None

    s3_client = get_s3_client()
    if not s3_client:
        return None

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    object_key = f"{region}/{flyover}/feedback/{timestamp}-{uuid4().hex}.json"

    payload = {
        "region": region,
        "flyover": flyover,
        "generated_at": timestamp,
        "source": "dashboard",
        "records": records,
    }

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key,
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        return f"s3://{S3_BUCKET_NAME}/{object_key}"
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error al subir feedback a S3: {exc}")
        return None
