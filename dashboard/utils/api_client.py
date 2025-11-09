"""Cliente de API para interactuar con el backend de FastAPI.

Este módulo proporciona funciones para realizar peticiones a los diferentes
endpoints de la API, encapsulando la lógica de `requests` y el manejo
básico de errores.
"""

import os
from typing import Any, cast

import requests
import streamlit as st

# Lee la URL de la API desde una variable de entorno, con un valor por defecto.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def get_regions() -> list[str]:
    """Obtiene la lista de regiones disponibles desde la API."""
    endpoint = f"{API_BASE_URL}/regions"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        # Le decimos a mypy que confiamos en que esto es una lista de strings
        return cast(list[str], response.json().get("regions", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener regiones: {e}")
        return []


def get_flyovers(region: str) -> list[str]:
    """Obtiene la lista de sobrevuelos para una región específica."""
    endpoint = f"{API_BASE_URL}/flyovers/{region}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        # Le decimos a mypy que confiamos en que esto es una lista de strings
        return cast(list[str], response.json().get("flyovers", []))
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener sobrevuelos para {region}: {e}")
        return []


def get_detection_results(region: str, flyover: str) -> dict[str, Any]:
    """Obtiene los resultados de detección para un sobrevuelo específico."""
    endpoint = f"{API_BASE_URL}/results/{region}/{flyover}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener resultados para {region}/{flyover}: {e}")
        return {"error": str(e)}


def process_images(image_urls: list[str]) -> dict[str, Any]:
    """Llama al endpoint de la API para procesar una lista de URLs de imágenes.

    Args:
        image_urls: Una lista de URLs de S3 correspondientes a las imágenes a procesar.

    Returns:
        Un diccionario con la respuesta de la API, que puede contener los resultados
        del procesamiento o un mensaje de error.
    """
    endpoint = f"{API_BASE_URL}/predict-many"
    payload = {"urls": image_urls}

    try:
        response = requests.post(endpoint, json=payload, timeout=300)  # Timeout extendido
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error al llamar a la API de procesamiento: {e}")
        return {"error": str(e)}


def get_counts_for_flyover(region: str, flyover: str) -> dict[str, Any]:
    """Obtiene los conteos de un sobrevuelo específico."""
    endpoint = f"{API_BASE_URL}/counts/{region}/{flyover}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener conteos para {region}/{flyover}: {e}")
        return {"error": str(e)}


def get_counts_for_region(region: str) -> dict[str, Any]:
    """Obtiene los conteos agregados para una región específica."""
    endpoint = f"{API_BASE_URL}/counts/{region}"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener conteos para {region}: {e}")
        return {"error": str(e)}
