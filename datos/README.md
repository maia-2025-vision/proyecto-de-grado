# Datos (rúbrica)

Esta carpeta existe para cumplir la rúbrica: aquí dejamos referencias claras a los datos y algunas muestras mínimas. Los datos completos (train/val/test, groundtruth, parches) siguen versionados con DVC en `data/`.

## ¿Dónde están los datos completos?
- Requiere credenciales AWS (`AWS_PROFILE=dvc-user`).
- Comandos mínimos:
  ```bash
  aws configure --profile dvc-user  # solo una vez
  dvc pull data/groundtruth data/train data/val data/test
  dvc pull data/patches-512-ol-160-m01
  ```
- Más rutas opcionales están en `dvc.yaml`.

## Muestras incluidas (livianas)
- Anotaciones (primeras 10 filas): `datos/muestras/anotaciones_sample.csv`.
- Imágenes de entrenamiento (reducidas a máx 640px): `datos/muestras/imagenes/train/*.jpg`.
- Parches de prueba (reducidos): `datos/muestras/imagenes/patches/*.jpg`.

## Notas para calificación
- No subimos datasets completos ni modelos pesados al repo; se bajan con DVC/S3.
- Para reproducir pipelines o entrenar, ejecutar los `dvc pull` anteriores antes de usar `poe` o `dvc repro`.
