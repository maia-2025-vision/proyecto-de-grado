"""Página de Streamlit para la carga y procesamiento de imágenes.

Permite a los usuarios subir imágenes, especificar metadatos como la etiqueta
y la fecha, y enviar las imágenes a un bucket de S3 y a la API de procesamiento
para su análisis.
"""

import datetime

import streamlit as st

from dashboard.utils.api_client import process_images
from dashboard.utils.s3_utils import upload_files_to_s3

st.set_page_config(
    page_title="📤 Carga y Procesamiento de Imágenes",
    page_icon="📤",
    layout="wide",
)

st.title("📤 Carga y Procesamiento de Imágenes")

# --- UI Components ---
uploaded_files = st.file_uploader(
    "Seleccione una o más imágenes",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Arrastra y suelta archivos aquí.",
)

with st.sidebar:
    st.header("Parámetros de Detección")

    farm_name = st.text_input(
        "Región",
        value="Prueba",
        help="Una etiqueta para organizar este grupo de imágenes (ej: 'Parque_Kruger_Norte').",
    )

    col1, col2 = st.columns(2)
    with col1:
        capture_date = st.date_input("Fecha de captura", value=datetime.date.today())
    with col2:
        capture_time = st.time_input("Hora de captura", value=datetime.datetime.now().time())

process_button = st.button("Procesar Imágenes", disabled=not uploaded_files, type="primary")


# --- Logic ---

if process_button:
    # --- MODO API REAL ---
    if not farm_name:
        st.warning("Por favor, ingrese una 'Etiqueta' en la barra lateral.")
    else:
        flyover_datetime = datetime.datetime.combine(capture_date, capture_time)
        s3_urls = upload_files_to_s3(
            files=uploaded_files,
            region=farm_name,
            flyover_datetime=flyover_datetime,
        )

        if s3_urls:
            st.info(f"{len(s3_urls)} imágenes subidas. Enviando a la API para procesamiento...")

            with st.spinner(
                "La API está procesando las imágenes. Esto puede tardar varios minutos..."
            ):
                results = process_images(image_urls=s3_urls)

            if "error" in results:
                st.error(f"Error de la API: {results['error']}")
            else:
                st.success("¡Procesamiento completado!")
                st.info("Ahora puedes ir a 'Visualizador y Métricas' para ver los resultados.")
                with st.expander("Ver resumen de resultados de la API"):
                    st.json(results)
