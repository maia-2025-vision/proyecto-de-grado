import streamlit as st

# Define pages
home_page = st.Page("pages/0_Home.py", title="Página de Inicio", icon="🏠")
img_upload_page = st.Page(
    "pages/1_Upload_Images.py", title="Carga y Procesamiento de Imágenes", icon="📤"
)
detection_viewer_page = st.Page(
    "pages/2_Detection_Viewer.py", title="Visualización de Detecciones", icon="🖼️"
)
metrics_viewer_page = st.Page("pages/3_Metrics_Viewer.py", title="Métricas de Detección", icon="📊")

# Create navigation
pg = st.navigation(
    {
        "Páginas": [home_page, img_upload_page, detection_viewer_page, metrics_viewer_page],
    }
)

# Run the selected page
pg.run()
