import pandas as pd
import streamlit as st
from PIL import Image

from dashboard.utils.api_client import (
    get_counts_for_flyover,
    get_detection_results,
    get_flyovers,
    get_regions,
)
from dashboard.utils.visualization import SPECIES_MAP, download_image, draw_detections_on_image

st.set_page_config(page_title="Resultados de Detección", layout="wide")

st.title("Visualizador y Métricas de Detección")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Selección de Datos")
    regions = get_regions()
    if not regions:
        st.warning("No se pudieron cargar las regiones.")
        selected_region = None
    else:
        selected_region = st.selectbox("Región", options=regions)

    if selected_region:
        flyovers = get_flyovers(selected_region)
        if not flyovers:
            st.warning("No hay sobrevuelos para esta región.")
            selected_flyover = None
        else:
            selected_flyover = st.selectbox("Sobrevuelo", options=flyovers)
    else:
        selected_flyover = None

    if st.button("Cargar Resultados", disabled=not selected_flyover, type="primary"):
        with st.spinner("Cargando resultados y conteos..."):
            # Cargar ambos, detecciones y conteos
            det_results = get_detection_results(selected_region, selected_flyover)
            count_results = get_counts_for_flyover(selected_region, selected_flyover)

            if "error" in det_results:
                st.error(f"Error al cargar detecciones: {det_results['error']}")
                st.session_state.detection_results = {}
            else:
                st.session_state.detection_results = det_results

            if "error" in count_results:
                st.error(f"Error al cargar conteos: {count_results['error']}")
                st.session_state.count_results = {}
            else:
                st.session_state.count_results = count_results

    st.markdown("---")
    st.header("Opciones de Visualización")
    confidence_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)
    line_width = st.slider("Grosor de Línea", 1, 10, 3)
    text_color = st.color_picker("Color del Texto", "#FFFFFF")

    st.header("Filtro por Especie")
    if "detection_results" in st.session_state and st.session_state.detection_results:
        labels_in_results = set()
        for res in st.session_state.detection_results.get("results", []):
            labels_in_results.update(res.get("detections", {}).get("labels", []))

        species_options = {
            label: SPECIES_MAP.get(label, "Unknown") for label in sorted(labels_in_results)
        }
        selected_species_names = st.multiselect(
            "Especies a mostrar:",
            options=list(species_options.values()),
            default=list(species_options.values()),
        )
        selected_labels = [
            label for label, name in species_options.items() if name in selected_species_names
        ]
    else:
        st.info("Cargue resultados para ver filtros.")
        selected_labels = []

# --- Main Display Area ---

# 1. Metrics and Counts Section
if "count_results" in st.session_state and st.session_state.count_results:
    st.subheader(f"Métricas para: {selected_region} / {selected_flyover}")
    counts_data = st.session_state.count_results.get("total_counts", {})

    if not counts_data:
        st.warning("No hay datos de conteo para este sobrevuelo.")
    else:
        # Mapear IDs de especies a nombres
        named_counts = {SPECIES_MAP.get(int(k), f"ID_{k}"): v for k, v in counts_data.items()}
        df_counts = pd.DataFrame(list(named_counts.items()), columns=["Especie", "Conteo"])

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_counts)
        with col2:
            st.bar_chart(df_counts.set_index("Especie"))

st.markdown("---")

# 2. Detections Visualization Section
if "detection_results" in st.session_state and st.session_state.detection_results:
    st.subheader("Visualización de Detecciones")
    results_data = st.session_state.detection_results.get("results", [])
    image_urls = [res["url"] for res in results_data if "url" in res]

    if not image_urls:
        st.warning("No se encontraron imágenes en los resultados de este sobrevuelo.")
    else:
        selected_image_url = st.selectbox(
            "Selecciona una imagen para visualizar:",
            image_urls,
            format_func=lambda url: url.split("/")[-1],
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Modelo: Cajas Delimitadoras")
            selected_result = next(
                (res for res in results_data if res.get("url") == selected_image_url), None
            )

            if selected_result and selected_image_url:
                image = download_image(selected_image_url)
                if image:
                    image_with_detections = draw_detections_on_image(
                        image.copy(),
                        selected_result.get("detections", {}),
                        confidence_threshold,
                        selected_labels,
                        text_color=text_color,
                        line_width=line_width,
                    )
                    st.image(
                        image_with_detections,
                        caption="Detecciones con Faster R-CNN",
                        use_container_width=True,
                    )

        with col2:
            st.subheader("Modelo: Mapa de Densidad (Pendiente)")
            st.image(
                Image.new("RGB", (800, 600), color="darkgrey"),
                caption="Visualización de HerdNet",
                use_container_width=True,
            )
else:
    st.info(
        "👈 Selecciona una región y sobrevuelo y haz clic en 'Cargar Resultados' para comenzar."
    )
