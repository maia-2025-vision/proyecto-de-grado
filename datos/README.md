# Datos (rúbrica)

Esta carpeta existe para cumplir la rúbrica: aquí dejamos referencias claras a los datos. Los datos completos (train/val/test, groundtruth, parches) siguen versionados con DVC en `data/`.

## Datos
- Muestra de datos en carpeta de OneDrive:  
https://uniandes-my.sharepoint.com/:f:/g/personal/m_restrepom2_uniandes_edu_co/IgDGZxNV1Lo5R6YWVQ4kV2mJAf48dHXvYutq1Q-A7_xIHn4?e=0YVPpA

- Requiere credenciales AWS (`AWS_PROFILE=dvc-user`).
- Comandos mínimos:
  ```bash
  aws configure --profile dvc-user  # solo una vez
  dvc pull data/groundtruth data/train data/val data/test
  dvc pull data/patches-512-ol-160-m01
  ```
- Más rutas opcionales están en `dvc.yaml`.

## Notas para calificación
- No subimos datasets completos ni modelos pesados al repo; se bajan con DVC/S3.
- Para reproducir pipelines o entrenar, ejecutar los `dvc pull` anteriores antes de usar `poe` o `dvc repro`.
