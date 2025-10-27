import datetime

import streamlit as st

from dashboard.utils.api_client import process_images
from dashboard.utils.local_detector import run_local_detection
from dashboard.utils.s3_utils import upload_files_to_s3

st.set_page_config(
    page_title="Carga y Procesamiento de Imágenes",
    page_icon="📤",
    layout="wide",
)

st.title("Carga y Procesamiento de Imágenes")

# --- UI Components ---
with st.sidebar:
    st.header("Modo de Operación")
    use_local_mock = st.toggle(
        "Usar Datos de Ejemplo",
        value=False,  # Por defecto desactivado para priorizar el flujo real
        help="Si está activado, usará un conjunto de datos predefinido en lugar de la API real.",
    )
    st.markdown("---")

if use_local_mock:
    st.info("Se cargará una imagen de ejemplo con resultados predefinidos para la demostración.")
    process_button = st.button("Cargar Ejemplo", type="primary")

else:
    uploaded_files = st.file_uploader(
        "Seleccione una o más imágenes",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Arrastra y suelta archivos aquí.",
    )
    with st.sidebar:
        st.header("Parámetros de Detección")

        farm_name = st.text_input(
            "Etiqueta",
            value="Finca_Ejemplo",
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
    if use_local_mock:
        # --- MODO SIMULADO ---
        st.info("Cargando datos de ejemplo...")
        image, mock_results = run_local_detection()

        if image and mock_results:
            st.session_state.detection_results = mock_results
            image_url_key = mock_results["results"][0]["url"]
            st.session_state.original_images = {image_url_key: image}

            total_counts = {}
            for res in mock_results.get("results", []):
                for count_data in res.get("counts_by_species", []):
                    for label, num in count_data.get("counts", {}).items():
                        total_counts[label] = total_counts.get(label, 0) + num
            st.session_state.count_results = {"total_counts": total_counts}

            st.success("¡Datos de ejemplo cargados!")
            st.info("Ahora puedes ir a 'Visualizador y Métricas' para ver los resultados.")
            st.image(image, caption="Imagen de ejemplo cargada", width=300)
        else:
            st.error("No se pudieron cargar los datos de ejemplo. Revisa los logs.")

    else:
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
