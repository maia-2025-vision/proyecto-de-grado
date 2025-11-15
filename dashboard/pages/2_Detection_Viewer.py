"""Página de Streamlit para visualizar los resultados de las detecciones."""

from typing import TypeVar

import streamlit as st
from loguru import logger
from streamlit.delta_generator import DeltaGenerator

from dashboard.utils.api_client import (
    FlyoverResults,
    ImageResults,
    get_detection_results,
    get_flyovers,
    get_regions,
)
from dashboard.utils.visualization import (
    SPECIES_MAP,
    AnnotParams,
    download_image,
    draw_boxes_on_image,
    draw_points_on_image,
    render_summary_tables,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# DetectionState = dict[str, Any]
# if "detection_state" not in st.session_state:
#    st.session_state["detection_state"] = None

# def store_detections(state: DetectionState | None) -> None:
#    """Persist detection payload + metadata for later renders."""
#    st.session_state["detection_state"] = state

# current_state = cast(DetectionState | None, st.session_state.get("detection_state"))

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

T_ = TypeVar("T_")


def safe_find(a_list: list[T_], value: T_, default: int = 0) -> int:
    index = a_list.index(value)
    if index < 0:
        return default
    else:
        return index


def side_bar() -> tuple[FlyoverResults, AnnotParams]:
    with st.sidebar:
        st.header("Selección de Datos")
        regions: list[str] = get_regions()

        if len(regions) == 0:
            st.warning("No se pudieron cargar las regiones.")
            selected_region = None
        else:
            region_param = st.query_params.get("region")
            reg_index = 0
            if region_param is not None:
                reg_index = safe_find(regions, region_param)
                logger.info(f"Flyover from url: {region_param}, fo_index: {reg_index}")

            selected_region = st.selectbox("Región", options=regions, index=reg_index)

        if selected_region is not None:
            flyovers = get_flyovers(selected_region)
            if not flyovers:
                st.warning("No hay sobrevuelos para esta región.")
                selected_flyover = None
            else:
                flyover_param = st.query_params.get("flyover")
                fo_index = 0
                if flyover_param is not None:
                    fo_index = safe_find(flyovers, flyover_param)
                    logger.info(f"Flyover from url: {flyover_param}, fo_index: {fo_index}")

                selected_flyover = st.selectbox("Sobrevuelo", options=flyovers, index=fo_index)
        else:
            selected_flyover = None

        # Carga automática sin botón simplifica el flujo de datos:
        if selected_region is not None and selected_flyover is not None:
            with st.spinner("Cargando resultados de detección..."):
                det_results = get_detection_results(selected_region, selected_flyover)
                num_results = len(det_results.results_by_image)
                logger.info(f"se cargaron detecciones para {num_results} imágenes")

            if det_results.error is not None:
                st.error(f"Error al cargar detecciones: {det_results.error}")
        else:  # default value
            det_results = FlyoverResults(
                region="__undefined__",
                flyover="__undefined__",
                results_by_image={},
                error="det_results is still uninitialized",
            )

        st.markdown("---")
        st.header("Opciones de Visualización")
        confidence_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.5, 0.05)
        line_width = st.slider("Grosor de Líneas", 1, 10, 1)
        point_size = st.slider("Tamaño de Anotación", 1, 20, 12)
        add_text_boxes = st.checkbox("Añadir etiquetas Textuales", value=False)
        text_color = st.color_picker("Color del Texto", "#FFFFFF")

        st.header("Filtro por Especie")

        if len(det_results.results_by_image) > 0:
            labels_in_results = set()
            for img_result in det_results.results_by_image.values():
                labels_in_results |= set(img_result.detections.labels)

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

        return det_results, AnnotParams(
            confidence_threshold=confidence_threshold,
            text_color=text_color,
            point_size=point_size,
            line_width=line_width,
            selected_labels=selected_labels,
            add_text_boxes=add_text_boxes,
        )


def draw_img_and_annots(
    image_results: ImageResults, annot_params: AnnotParams, mode_placeholder: DeltaGenerator
) -> None:
    image_url = image_results.url

    image = download_image(image_url)
    if not image:
        st.error("No fue posible descargar la imagen seleccionada.")
    else:
        detections = image_results.detections
        has_boxes = bool(detections.boxes)
        has_points = bool(detections.points) or has_boxes

        available_modes: list[str] = []
        if has_boxes:
            available_modes.append("Cajas")
        if has_points:
            available_modes.append("Puntos")

        if len(available_modes) == 0:
            st.warning("Este resultado no contiene geometrías para visualizar.")
        else:
            display_mode = mode_placeholder.radio(
                "Modo",
                options=available_modes,
                horizontal=True,
                index=0,
                key=f"mode-{image_url}",
            )

            if display_mode == "Cajas":
                st.subheader("Modelo: Cajas Delimitadoras")
                rendered_image = draw_boxes_on_image(image.copy(), detections, annot_params)
                caption = "Detecciones con Cajas Delimitadoras"
            else:
                st.subheader("Modelo: Puntos")
                rendered_image = draw_points_on_image(image.copy(), detections, annot_params)
                caption = "Detecciones por Puntos"

            image_width = image.size[0]
            render_width = max(image_width, 768)
            st.image(rendered_image, caption=caption, width=render_width)


# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------


def render_page() -> None:
    st.set_page_config(page_title="Visualizador de Detecciones", page_icon="🖼️", layout="wide")
    st.title("🖼️ Visualizador de Detecciones")

    det_results, annot_params = side_bar()

    if len(det_results.results_by_image):
        region = det_results.region
        flyover = det_results.flyover
        results_data = det_results.results_by_image
        image_fnames = list(results_data.keys())

        if len(image_fnames) == 0:
            st.warning("No se encontraron imágenes en los resultados de este sobrevuelo.")
        else:
            st.subheader(f"Detecciones para: {region} / {flyover}")

            col_img, col_mode = st.columns([3, 1])
            with col_img:
                image_param = st.query_params.get("image")
                logger.info(f"Image from url: {image_param}")
                if isinstance(image_param, str):
                    img_idx = safe_find(image_fnames, image_param)
                else:
                    img_idx = 0

                selected_image_fname = st.selectbox(
                    "Imagen a visualizar", image_fnames, index=img_idx
                )
            # logger.info(f"{selected_image_url=}")
            mode_placeholder = col_mode.empty()

            col_img, col_summary = st.columns([3, 1])

            img_results = det_results.results_by_image[selected_image_fname]
            with col_img:
                draw_img_and_annots(img_results, annot_params, mode_placeholder)
            with col_summary:
                render_summary_tables(img_results, annot_params)

    else:
        st.info("👈 Selecciona una región y un sobrevuelo.")
        st.info("Luego presiona 'Cargar Detecciones' para comenzar.")


if __name__ == "__main__":
    render_page()
