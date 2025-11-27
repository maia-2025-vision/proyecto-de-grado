# Modelos 


## Pesos mínimos (DVC)


```bash
aws configure --profile dvc-user  # credenciales fueron compartidas en el documento de entrega  
dvc pull data/models/herdnet_v2_hn2/best_model.pth
dvc pull data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth
```

## Enlaces a archivos con pesos (.pth)
- Carpeta compartida (HerdNet v2 HN y Faster R-CNN ResNet50):
  https://uniandes-my.sharepoint.com/:f:/g/personal/m_restrepom2_uniandes_edu_co/IgDmPNnFNTkpSq_Dc8qcatbyAZFl3EskTaDTrUyM6mI-xdI?e=MtBuIv


## Rutas usadas por la API
- HerdNet: `MODEL_WEIGHTS_PATH=data/models/herdnet_v2_hn2/best_model.pth` (ver `pyproject.toml`, tarea `serve-hn`).
- Faster R-CNN: `MODEL_WEIGHTS_PATH=data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth` (tarea `serve-frc`).
