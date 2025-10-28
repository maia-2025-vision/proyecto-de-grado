"""Página principal de la aplicación de Streamlit."""

import streamlit as st

st.set_page_config(
    page_title="Detección de Animales en Imágenes Aéreas",
    page_icon="🐘",
    layout="wide",
)

st.title("Aplicación de Conteo y Detección de Animales")

st.markdown(
    """
    Bienvenido a la aplicación para el conteo y detección de animales en manadas
    densas a partir de imágenes aéreas.

    **Seleccione una opción de la barra lateral para comenzar.**
    """
)

st.info("Seleccione una página.", icon="👈")
