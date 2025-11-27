# Modelos 



## Pesos mínimos (DVC)
```bash
aws configure --profile dvc-user  # si no lo has hecho
dvc pull data/models/herdnet_v2_hn2/best_model.pth
dvc pull data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth
```

## Enlaces externos (OneDrive)
- HerdNet v2 HN2: [Enlace OneDrive:](https://drive.google.com/drive/folders/1Jl_-KSijjS8c1iGWhpp9Db82yyTDkic3)
- Faster R-CNN (ResNet50): [Enlace OneDrive:](https://drive.google.com/drive/folders/1Jl_-KSijjS8c1iGWhpp9Db82yyTDkic3)



## Rutas usadas por la API
- HerdNet: `MODEL_WEIGHTS_PATH=data/models/herdnet_v2_hn2/best_model.pth` (ver `pyproject.toml`, tarea `serve-hn`).
- Faster R-CNN: `MODEL_WEIGHTS_PATH=data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth` (tarea `serve-frc`).
