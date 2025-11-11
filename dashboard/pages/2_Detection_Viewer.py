"""Página de Streamlit para visualizar los resultados de las detecciones."""

import streamlit as st

from dashboard.utils.api_client import (
    get_detection_results,
    get_flyovers,
    get_regions,
)
from dashboard.utils.visualization import (
    SPECIES_MAP,
    download_image,
    draw_centroids_on_image,
    draw_detections_on_image,
)

st.set_page_config(page_title="Visualizador de Detecciones", page_icon="🖼️", layout="wide")

st.title("🖼️ Visualizador de Detecciones")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

if "detection_state" not in st.session_state:
    st.session_state.detection_state = None


def store_detections(state: dict | None) -> None:
    """Persist detection payload + metadata for later renders."""
    st.session_state.detection_state = state


current_state: dict | None = st.session_state.detection_state

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
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

    load_button_disabled = not (selected_region and selected_flyover)
    if st.button("Cargar Detecciones", disabled=load_button_disabled, type="primary"):
        if selected_region and selected_flyover:
            with st.spinner("Cargando resultados de detección..."):
                det_results = get_detection_results(selected_region, selected_flyover)

            if "error" in det_results:
                st.error(f"Error al cargar detecciones: {det_results['error']}")
                store_detections(None)
            else:
                store_detections(
                    {
                        "payload": det_results,
                        "region": selected_region,
                        "flyover": selected_flyover,
                    }
                )
        else:
            store_detections(None)

    st.markdown("---")
    st.header("Opciones de Visualización")
    confidence_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)
    line_width = st.slider("Grosor de Línea (Cajas)", 1, 10, 3)
    point_size = st.slider("Tamaño de Punto (Centroides)", 1, 20, 5)
    text_color = st.color_picker("Color del Texto", "#FFFFFF")

    st.header("Filtro por Especie")
    if current_state and current_state["payload"].get("results"):
        labels_in_results = set()
        for res in current_state["payload"].get("results", []):
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
        st.info("Carga detecciones para habilitar los filtros.")
        selected_labels = []

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
if current_state and current_state["payload"].get("results"):
    payload = current_state["payload"]
    region = current_state["region"]
    flyover = current_state["flyover"]
    results_data = payload.get("results", [])
    image_urls = [res.get("url") for res in results_data if res.get("url")]

    if not image_urls:
        st.warning("No se encontraron imágenes en los resultados de este sobrevuelo.")
    else:
        st.subheader(f"Detecciones para: {region} / {flyover}")

        col_img, col_mode = st.columns([3, 1])
        with col_img:
            selected_image_url = st.selectbox(
                "Imagen a visualizar",
                image_urls,
                format_func=lambda url: url.split("/")[-1],
            )
        mode_placeholder = col_mode.empty()

        selected_result = next(
            (res for res in results_data if res.get("url") == selected_image_url), None
        )

        if selected_result:
            image = download_image(selected_image_url)

            if not image:
                st.error("No fue posible descargar la imagen seleccionada.")
            else:
                detections = selected_result.get("detections", {}) or {}
                has_boxes = bool(detections.get("boxes"))
                has_points = bool(detections.get("points")) or has_boxes

                available_modes: list[str] = []
                if has_boxes:
                    available_modes.append("Cajas")
                if has_points:
                    available_modes.append("Centroides")

                if not available_modes:
                    st.warning("Este resultado no contiene geometrías para visualizar.")
                else:
                    display_mode = mode_placeholder.radio(
                        "Modo",
                        options=available_modes,
                        horizontal=True,
                        index=0,
                        key=f"mode-{selected_image_url}",
                    )

                    if display_mode == "Cajas":
                        st.subheader("Modelo: Cajas Delimitadoras")
                        rendered_image = draw_detections_on_image(
                            image.copy(),
                            detections,
                            confidence_threshold,
                            selected_labels,
                            text_color=text_color,
                            line_width=line_width,
                        )
                        caption = "Detecciones con Cajas Delimitadoras"
                    else:
                        st.subheader("Modelo: Centroides")
                        rendered_image = draw_centroids_on_image(
                            image.copy(),
                            detections,
                            confidence_threshold,
                            selected_labels,
                            point_size=point_size,
                        )
                        caption = "Detecciones con Centroides"

                    st.image(rendered_image, caption=caption, use_container_width=True)
        else:
            st.error("No se encontró el resultado seleccionado.")
else:
    st.info("👈 Selecciona una región y un sobrevuelo.")
    st.info("Luego presiona 'Cargar Detecciones' para comenzar.")
