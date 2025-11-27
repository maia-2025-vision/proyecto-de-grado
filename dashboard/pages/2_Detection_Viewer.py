"""Página de Streamlit para visualizar los resultados de las detecciones."""

from typing import Any, TypeVar

import streamlit as st
from loguru import logger
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from api.schemas.req_resp_types import Detections
from dashboard.utils.api_client import (
    FlyoverResults,
    ImageResults,
    get_detection_results,
    get_flyovers,
    get_regions,
)
from dashboard.utils.s3_utils import upload_feedback_payload
from dashboard.utils.visualization import (
    SPECIES_MAP,
    AnnotParams,
    create_thumbnail_with_marker,
    download_image,
    draw_boxes_on_image,
    draw_points_on_image,
    render_summary_tables,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

SPECIES_NAME_OPTIONS = [SPECIES_MAP[idx] for idx in sorted(SPECIES_MAP.keys())]
SPECIES_ID_BY_NAME = {name: idx for idx, name in SPECIES_MAP.items()}
FEEDBACK_STATE_KEY = "detection_feedback_buffer"
THUMBNAIL_COLUMNS = 5


def ensure_feedback_state() -> dict[str, dict[str, object]]:
    """Devuelve (y crea si no existe) el buffer de cambios en sesión."""
    return st.session_state.setdefault(FEEDBACK_STATE_KEY, {})  # type: ignore[no-any-return]


def update_feedback_entry(entry_key: str, metadata: dict[str, Any], new_label_name: str) -> None:
    """Agrega o elimina una reclasificación del buffer."""
    feedback_state = ensure_feedback_state()
    new_label_id = SPECIES_ID_BY_NAME.get(new_label_name, metadata["original_label"])
    if new_label_id == metadata["original_label"]:
        feedback_state.pop(entry_key, None)
    else:
        feedback_state[entry_key] = metadata | {
            "new_label": new_label_id,
            "new_label_name": new_label_name,
        }


def build_detection_entries(
    *,
    image: Image.Image,
    detections: Detections,
    crop_size: int,
    thumb_size: int,
) -> list[dict[str, Any]]:
    """Genera los parches que se mostrarán en el grid de feedback."""
    entries: list[dict[str, Any]] = []
    boxes = detections.boxes or []
    points = detections.points or []
    labels = detections.labels or []
    scores = detections.scores or []

    num_detections = len(labels) or len(points) or len(boxes)
    if num_detections == 0:
        return entries

    width, height = image.size

    for idx in range(num_detections):
        label = labels[idx] if idx < len(labels) else 0
        score = scores[idx] if idx < len(scores) else None

        if boxes and idx < len(boxes) and boxes[idx] is not None:
            xmin_box, ymin_box, xmax_box, ymax_box = boxes[idx]
            center_x = (xmin_box + xmax_box) / 2
            center_y = (ymin_box + ymax_box) / 2
            bbox = [xmin_box, ymin_box, xmax_box, ymax_box]
        elif points and idx < len(points) and points[idx] is not None:
            center_x, center_y = points[idx]
            bbox = None
        else:
            continue

        half_patch = crop_size / 2
        xmin = max(0, int(center_x - half_patch))
        ymin = max(0, int(center_y - half_patch))
        xmax = min(width, int(center_x + half_patch))
        ymax = min(height, int(center_y + half_patch))

        if xmin >= xmax or ymin >= ymax:
            continue

        # Crear thumbnail a partir del recorte alrededor de la detección
        crop = image.crop((xmin, ymin, xmax, ymax))
        marker_center = (int(center_x - xmin), int(center_y - ymin))
        thumbnail = create_thumbnail_with_marker(crop, marker_center, size=thumb_size)
        entries.append(
            {
                "index": idx,
                "label_id": label,
                "label_name": SPECIES_MAP.get(label, f"Label {label}"),
                "score": score,
                "patch": thumbnail,
                "center": (center_x, center_y),
                "bbox": bbox,
            }
        )

    return entries


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
        point_size = st.slider("Tamaño de Anotación", 1, 20, 2)
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
    image_results: ImageResults,
    annot_params: AnnotParams,
    mode_placeholder: DeltaGenerator,
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


def render_feedback_panel(
    *,
    region: str,
    flyover: str,
    image_name: str,
    image_results: ImageResults,
    base_image: Image.Image | None,
) -> None:
    """Renderiza el grid de thumbnails para reclasificación."""
    st.markdown("---")
    st.subheader("📝 Feedback de detecciones")

    if base_image is None:
        base_image = download_image(image_results.url)
    if base_image is None:
        st.info("No fue posible descargar la imagen para generar thumbnails.")
        return

    max_detections = st.slider(
        "Máximo de detecciones a mostrar",
        min_value=4,
        max_value=60,
        value=24,
        step=4,
        key=f"feedback-limit-{region}-{flyover}",
    )

    zoom_pct = st.slider(
        "Zoom del recorte (%)",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
        key=f"feedback-zoom-{region}-{flyover}",
    )

    crop_size = st.slider(
        "Tamaño de recorte (px)",
        min_value=64,
        max_value=256,
        value=160,
        step=16,
        key=f"feedback-crop-{region}-{flyover}",
    )
    thumb_size = max(32, int(crop_size * zoom_pct / 100))

    detection_entries = build_detection_entries(
        image=base_image,
        detections=image_results.detections,
        crop_size=crop_size,
        thumb_size=thumb_size,
    )
    if not detection_entries:
        st.info("No hay detecciones disponibles para generar thumbnails.")
        return

    detection_entries = detection_entries[:max_detections]
    feedback_state = ensure_feedback_state()

    st.caption("Selecciona una etiqueta distinta para añadirla al lote de feedback.")
    for start in range(0, len(detection_entries), THUMBNAIL_COLUMNS):
        cols = st.columns(THUMBNAIL_COLUMNS)
        for col, entry in zip(
            cols, detection_entries[start : start + THUMBNAIL_COLUMNS], strict=False
        ):
            with col:
                caption = entry["label_name"]
                if entry["score"] is not None:
                    caption = f"{caption} ({entry['score']:.2f})"
                st.image(entry["patch"], caption=caption, width=thumb_size)

                entry_key = f"{region}|{flyover}|{image_name}|{entry['index']}"
                default_idx = (
                    SPECIES_NAME_OPTIONS.index(entry["label_name"])
                    if entry["label_name"] in SPECIES_NAME_OPTIONS
                    else 0
                )
                new_label = st.selectbox(
                    "Etiqueta",
                    SPECIES_NAME_OPTIONS,
                    index=default_idx,
                    key=f"{entry_key}-select",
                    label_visibility="collapsed",
                )
                metadata = {
                    "region": region,
                    "flyover": flyover,
                    "image": image_name,
                    "image_url": image_results.url,
                    "detection_index": entry["index"],
                    "original_label": entry["label_id"],
                    "original_label_name": entry["label_name"],
                    "score": entry["score"],
                    "center": list(entry["center"]),
                    "bbox": list(entry["bbox"]) if entry["bbox"] is not None else None,
                }
                update_feedback_entry(entry_key, metadata, new_label)

    pending_entries = [
        entry
        for entry in feedback_state.values()
        if entry["region"] == region and entry["flyover"] == flyover
    ]
    if pending_entries:
        st.success(f"{len(pending_entries)} reclasificaciones listas para subir.")
        if st.button("Subir feedback a S3", type="primary"):
            payload = []
            for entry in pending_entries:
                item = entry.copy()

                for key in ["patch", "index", "label_id", "label_name"]:
                    item.pop(key, None)
                payload.append(item)
            s3_uri = upload_feedback_payload(region=region, flyover=flyover, records=payload)
            if s3_uri:
                st.success(f"Feedback subido correctamente a {s3_uri}")
                for key in list(feedback_state.keys()):
                    data = feedback_state[key]
                    if data["region"] == region and data["flyover"] == flyover:
                        feedback_state.pop(key, None)
                        # También limpiar el estado del widget para que se resetee visualmente
                        widget_key = f"{key}-select"
                        if widget_key in st.session_state:
                            del st.session_state[widget_key]
    else:
        st.info("Cuando cambies una etiqueta aparecerá aquí para subirla.")


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
            base_image = download_image(img_results.url)
            with col_img:
                draw_img_and_annots(img_results, annot_params, mode_placeholder)
            with col_summary:
                render_summary_tables(img_results, annot_params)

            render_feedback_panel(
                region=region,
                flyover=flyover,
                image_name=selected_image_fname,
                image_results=img_results,
                base_image=base_image,
            )

    else:
        st.info("👈 Selecciona una región y un sobrevuelo.")
        st.info("Luego presiona 'Cargar Detecciones' para comenzar.")


if __name__ == "__main__":
    render_page()
