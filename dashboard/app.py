import streamlit as st

# Define pages
home_page = st.Page("pages/0_Home.py", title="Página de Inicio", icon="🏠")
img_upload_page = st.Page(
    "pages/1_Upload_Images.py", title="Carga y Procesamiento de Imágenes", icon="📤"
)
view_dets_page = st.Page(
    "pages/2_View_Detections.py", title="Visualización de Detecciones", icon="🖼️"
)

# Create navigation
pg = st.navigation(
    {
        "Páginas": [home_page, img_upload_page, view_dets_page],
    }
)

# Run the selected page
pg.run()
