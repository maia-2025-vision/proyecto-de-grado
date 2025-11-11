"""Página de Streamlit para visualizar las métricas de las detecciones."""

import pandas as pd
import streamlit as st

from dashboard.utils.api_client import (
    get_counts_for_flyover,
    get_flyovers,
    get_regions,
)

st.set_page_config(page_title="Métricas de Detección", page_icon="📊", layout="wide")

st.title("📊 Métricas de Detección")

# -----------------------------------------------------------------------------
# Estado
# -----------------------------------------------------------------------------
if "metrics_state" not in st.session_state:
    st.session_state.metrics_state = None


def store_metrics(state: dict | None) -> None:
    st.session_state.metrics_state = state


current_metrics = st.session_state.metrics_state

# -----------------------------------------------------------------------------
# Controles laterales
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

    load_disabled = not (selected_region and selected_flyover)
    if st.button("Cargar Métricas", disabled=load_disabled, type="primary"):
        if selected_region and selected_flyover:
            with st.spinner("Cargando conteos..."):
                count_results = get_counts_for_flyover(selected_region, selected_flyover)

            if "error" in count_results:
                st.error(f"Error al cargar conteos: {count_results['error']}")
                store_metrics(None)
            else:
                store_metrics(
                    {
                        "payload": count_results,
                        "region": selected_region,
                        "flyover": selected_flyover,
                    }
                )
        else:
            store_metrics(None)

# -----------------------------------------------------------------------------
# Contenido principal
# -----------------------------------------------------------------------------
if current_metrics and current_metrics["payload"].get("total_counts") is not None:
    payload = current_metrics["payload"]
    region = current_metrics["region"]
    flyover = current_metrics["flyover"]

    st.subheader(f"Métricas para: {region} / {flyover}")
    counts_data: dict[str, int] = payload.get("total_counts", {}) or {}

    if not counts_data:
        st.warning("No hay datos de conteo para este sobrevuelo.")
    else:
        if " Buffalo" in counts_data:
            counts_data["Buffalo"] = counts_data.pop(" Buffalo")

        df_counts = pd.DataFrame(sorted(counts_data.items()), columns=["Especie", "Conteo"])

        col_table, col_chart = st.columns(2)
        with col_table:
            st.dataframe(df_counts, hide_index=True, use_container_width=True)
        with col_chart:
            st.bar_chart(df_counts.set_index("Especie"))

        per_image = payload.get("per_image_counts") or payload.get("image_counts")
        if per_image:
            st.markdown("---")
            st.subheader("Conteos por Imagen")
            df_per_image = pd.DataFrame(per_image)
            if not df_per_image.empty:
                st.dataframe(df_per_image, use_container_width=True, hide_index=True)
else:
    st.info(
        "👈 Selecciona una región y sobrevuelo, luego presiona "
        "'Cargar Métricas' para ver los conteos."
    )
