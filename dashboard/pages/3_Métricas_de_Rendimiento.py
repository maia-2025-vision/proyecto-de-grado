import pandas as pd
import streamlit as st

from dashboard.utils.api_client import (
    get_counts_for_flyover,
    get_counts_for_region,
    get_flyovers,
    get_regions,
)

st.set_page_config(
    page_title="Métricas de Rendimiento",
    page_icon="📊",
    layout="wide",
)

st.title("Métricas de Rendimiento y Conteos")

# Reutilizar la lógica de selección de la otra página
regions = get_regions()
if not regions:
    st.warning("No se encontraron regiones.")
else:
    selected_region = st.selectbox(
        "Selecciona una Región para el reporte", options=["Todas"] + regions
    )

    if selected_region:
        if selected_region == "Todas":
            st.info(
                "Funcionalidad para 'Todas' las regiones pendiente. "
                "Por favor, selecciona una región específica."
            )
        else:
            flyovers = get_flyovers(selected_region)
            if not flyovers:
                st.warning(f"No se encontraron sobrevuelos para la región '{selected_region}'.")
            else:
                selected_flyover = st.selectbox(
                    "Selecciona un Sobrevuelo (o 'Todos' para un consolidado regional)",
                    options=["Todos"] + flyovers,
                )

                if st.button("Generar Reporte de Conteos"):
                    if selected_flyover == "Todos":
                        with st.spinner(
                            f"Calculando métricas para la región '{selected_region}'..."
                        ):
                            data = get_counts_for_region(selected_region)

                        st.header(f"Conteos Consolidados para la Región: {selected_region}")
                        if data and "grand_totals" in data:
                            df = pd.DataFrame(
                                list(data["grand_totals"].items()),
                                columns=["Especie", "Conteo Total"],
                            )
                            st.dataframe(df)
                            st.bar_chart(df.set_index("Especie"))
                        else:
                            st.error("No se pudieron obtener los datos de conteo para la región.")

                    else:
                        with st.spinner(
                            f"Calculando métricas para el sobrevuelo '{selected_flyover}'..."
                        ):
                            data = get_counts_for_flyover(selected_region, selected_flyover)

                        st.header(f"Conteos para: {selected_region} / {selected_flyover}")
                        if data and "total_counts" in data:
                            df = pd.DataFrame(
                                list(data["total_counts"].items()), columns=["Especie", "Conteo"]
                            )
                            st.dataframe(df)
                            st.bar_chart(df.set_index("Especie"))
                        else:
                            st.error(
                                "No se pudieron obtener los datos de conteo para el sobrevuelo."
                            )

# Placeholder para las métricas formales de la propuesta
st.header("Métricas de Evaluación del Modelo (Pendiente)")
st.info(
    "Esta sección mostrará métricas como F1-Score, MAE, y RMSE de conteo, "
    "comparando las predicciones con un set de datos de validación (ground truth)."
)
col1, col2, col3 = st.columns(3)
col1.metric("F1-Score Promedio", "N/A")
col2.metric("MAE (Error Absoluto Medio)", "N/A")
col3.metric("RMSE (Error Cuadrático Medio)", "N/A")
