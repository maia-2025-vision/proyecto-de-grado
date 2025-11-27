"""Página de Streamlit para visualizar las métricas de las detecciones."""

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from loguru import logger

from dashboard.utils.api_client import (
    get_counts_for_flyover,
    get_counts_for_region,
    get_flyovers,
    get_regions,
)
from dashboard.utils.visualization import SPECIES_MAP

st.set_page_config(page_title="Conteos de Sobrevuelo", page_icon="📊", layout="wide")

st.title("📊 Conteos de Sobrevuelo")


count_results: dict[str, Any] = {}
region_counts: dict[str, Any] = {}
selected_region: str | None = None
selected_flyover: str | None = None


def build_region_summary_table(region_payload: dict[str, Any]) -> pd.DataFrame:
    """Convierte el payload de /counts/<region> en un dataframe amigable."""
    totals_by_flyover = region_payload.get("totals_by_flyover") or {}
    rows: list[dict[str, Any]] = []

    for flyover_name, species_counts in sorted(totals_by_flyover.items()):
        row: dict[str, Any] = {"Sobrevuelo": flyover_name}
        if isinstance(species_counts, dict):
            for species_name, count in species_counts.items():
                row[str(species_name)] = count
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).fillna(0)
    species_columns = [name for name in SPECIES_MAP.values() if name in df.columns]
    other_columns = [col for col in df.columns if col not in {"Sobrevuelo", *species_columns}]
    ordered_cols = ["Sobrevuelo"] + species_columns + other_columns
    return df[ordered_cols].sort_values(by="Sobrevuelo")


def render_region_summary(region: str, region_payload: dict[str, Any]) -> None:
    """Muestra la tabla agregada por sobrevuelo + gráfico temporal."""
    df_region = build_region_summary_table(region_payload)
    if df_region.empty:
        st.info("Aún no hay métricas agregadas para esta región.")
        return

    st.subheader(f"Conteos agregados por sobrevuelo — {region}")
    st.dataframe(df_region, hide_index=True, width="stretch")

    chart_df = df_region.melt("Sobrevuelo", var_name="Especie", value_name="Conteo")
    chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x=alt.X("Sobrevuelo", axis=alt.Axis(labelAngle=-45)),
            y="Conteo",
            color="Especie",
        )
    )
    st.altair_chart(chart, use_container_width=True)


def get_counts_by_image(region: str, flyover: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    rows_out: list[dict[str, object]] = []

    for row in rows:
        url = row["url"]
        assert isinstance(url, str)
        image_fname = url.split("/")[-1]
        # print(row.keys())
        counts_at_thresh = row["counts_at_threshold"]
        assert isinstance(counts_at_thresh, dict)
        row_out = counts_at_thresh["counts"].copy()
        row_out["Imagen"] = image_fname
        row_out["Visualize"] = (
            "/Detection_Viewer?region=" + region + "&flyover=" + flyover + "&image=" + image_fname
        )
        rows_out.append(row_out)

    out = pd.DataFrame(rows_out)
    cols_out = ["Imagen", "Visualize"] + [
        species for species in SPECIES_MAP.values() if species in out.columns
    ]

    return out[cols_out]


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

    if selected_region:
        with st.spinner("Cargando métricas de región..."):
            region_counts = get_counts_for_region(selected_region)
        if "error" in region_counts:
            st.error(f"Error al cargar métricas agregadas: {region_counts['error']}")
            region_counts = {}

    # load_disabled = not (selected_region and selected_flyover)
    # if st.button("Cargar Métricas", disabled=load_disabled, type="primary"):
    if selected_region and selected_flyover:
        with st.spinner("Cargando conteos..."):
            count_results = get_counts_for_flyover(selected_region, selected_flyover)

        if "error" in count_results:
            st.error(f"Error al cargar conteos: {count_results['error']}")

# -----------------------------------------------------------------------------
# Contenido principal
# -----------------------------------------------------------------------------

if selected_region:
    render_region_summary(selected_region, region_counts)
    st.markdown("---")

if len(count_results) > 0:
    logger.info(f"{count_results.keys()=}")
    # payload = count_results["payload"]
    region = count_results["region"]
    flyover = count_results["flyover"]

    st.subheader(f"Conteos para sobrevuelo: {region} / {flyover}")
    counts_data: dict[str, int] = count_results.get("total_counts", {})

    if not counts_data:
        st.warning("No hay datos de conteo para este sobrevuelo.")
    else:
        # Esto es necesario?
        # if " Buffalo" in counts_data:
        #    counts_data["Buffalo"] = counts_data.pop(" Buffalo")

        df_counts = pd.DataFrame(sorted(counts_data.items()), columns=["Especie", "Conteo"])

        col_table, col_chart = st.columns(2)
        with col_table:
            st.dataframe(df_counts, hide_index=True, width="stretch")
        with col_chart:
            st.bar_chart(df_counts.set_index("Especie"), horizontal=True)

        # per_image = payload.get("per_image_counts") or payload.get("image_counts")
        assert selected_region is not None
        per_image = get_counts_by_image(
            region=selected_region, flyover=flyover, rows=count_results["rows"]
        )
        st.markdown("---")
        st.subheader("Conteos por Imagen")
        df_per_image = pd.DataFrame(per_image)
        if not df_per_image.empty:
            st.dataframe(
                df_per_image,
                width="stretch",
                hide_index=True,
                column_config={
                    "Visualize": st.column_config.LinkColumn(
                        "Visualizar",  # Display name for the column header
                        help="Clic para visualizar las detecciones.",
                        display_text="Visualizar",
                    )
                },
            )
else:
    st.info("👈 Selecciona una región y un sobrevuelo.")
    st.info("Luego presiona 'Cargar Métricas' para ver los conteos.")
