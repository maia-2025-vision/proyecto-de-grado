"""Módulo para almacenar datos de prueba (mock data) para el desarrollo de la UI.

Estos datos se extrajeron de una ejecución real del modelo Faster R-CNN
(de `data/test_results/frcnn-resnet50/detections.csv`) para proporcionar
una simulación realista de la salida de la API.
"""

# Usaremos detecciones de una sola imagen para simplificar el mock.
MOCK_IMAGE_NAME = "01802f75da35434ab373569fffc1fd65a3417aef_158.JPG"
MOCK_IMAGE_PATH = (
    "data/patches-512-ol-160-m0.3/test/01802f75da35434ab373569fffc1fd65a3417aef_158.JPG"
)


def get_mock_detection_results():
    """Devuelve un diccionario con resultados de detección que imita la estructura

    de la respuesta del endpoint /results/{region}/{flyover} de la API.
    """
    # Se comenta el cuerpo de la función para desactivar la lógica de simulación.
    # Si se necesita para desarrollo, descomentar las siguientes líneas.
    """
    # Estos son los datos reales del archivo detections.csv para la imagen MOCK_IMAGE_NAME
    detections_data = {
        "boxes": [
            [324.648193359375, 401.658935546875, 375.136474609375, 462.109130859375],
            [448.231689453125, 284.7304382324219, 493.1910400390625, 332.9855041503906],
            [333.8160095214844, 293.6655578613281, 381.8199157714844, 351.1039733886719],
            [406.9389953613281, 327.5141906738281, 470.7449645996094, 404.2540588378906],
            [329.5606384277344, 290.5695495605469, 361.6665954589844, 341.1286315917969],
            [13.990137100219727, 125.55936431884766, 53.726295471191406, 190.0941162109375],
            [422.7313232421875, 184.9056854248047, 479.26556396484375, 254.6704864501953],
            [294.102783203125, 359.2535400390625, 334.7113037109375, 420.46038818359375],
            [377.0632019042969, 316.4639587402344, 395.0378723144531, 345.4331359863281],
        ],
        "labels": [6, 6, 6, 6, 6, 6, 6, 6, 6],
        "scores": [
            0.9990876913070679,
            0.9986642599105835,
            0.9918422698974609,
            0.9768425226211548,
            0.8955750465393066,
            0.6551175713539124,
            0.6496358513832092,
            0.5950679183006287,
            0.3385568857192993,
        ],
    }

    # Estructura de respuesta completa
    api_response = {
        "results": [
            {
                "url": f"mock_data/{MOCK_IMAGE_NAME}",  # Usamos una URL falsa
                "detections": detections_data,
                "counts_by_species": [
                    {
                        "score_thresh": 0.0,
                        "counts": {"6": 9},  # 9 elefantes
                    }
                ],
            }
        ]
    }
    return api_response
    """
    return {}
