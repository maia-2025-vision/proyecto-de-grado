from PIL import Image

# from dashboard.utils.mock_data import get_mock_detection_results, MOCK_IMAGE_PATH


def run_local_detection() -> tuple[Image.Image | None, dict]:
    """Carga la imagen de prueba y los resultados de detección predefinidos

    desde el módulo de mock_data.
    """
    # Se comenta el cuerpo de la función para desactivar la lógica de simulación.
    """
    try:
        # Carga la imagen de ejemplo desde su ruta en el repo
        image = Image.open(MOCK_IMAGE_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen de prueba en '{MOCK_IMAGE_PATH}'")
        return None, {}

    # Obtiene los datos de detección predefinidos
    detection_results = get_mock_detection_results()

    return image, detection_results
    """
    return None, {}
