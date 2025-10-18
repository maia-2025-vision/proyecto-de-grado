# HerdNet - Entrenamiento y Uso

## Instalación de Datos

Antes de comenzar el entrenamiento, es necesario obtener los datos utilizando DVC:

```bash
# Obtener los parches de imágenes para entrenamiento
uv run dvc pull patches_512.dvc

# Obtener los modelos pre-entrenados (si se requiere)
uv run dvc pull models.dvc
```

## Configuración

La configuración del entrenamiento se encuentra en el archivo `oficial_herdnet/scripts/train.yml`. Este archivo contiene los hiperparámetros del modelo, rutas a los datos, y configuración general del entrenamiento:

```yaml
hyperparameters:
  epochs: 1
  batch_size: 8
  optimizer: "adam"
  learning_rate: 1e-4
  weight_decay: 0.0005

trainer:
  csv_logger: True
  paths:
    train_csv: "data/patches_512/gt/points_train_gt.csv"
    val_csv: "data/patches_512/gt/points_val_gt.csv"
    train_root: "data/patches_512/train"
    val_root: "data/patches_512/val"
    work_dir: "data/models/herdnet_v1"

model:
  pretrained: True
  num_classes: 7
  down_ratio: 2
```

Puede modificar estos parámetros según sus necesidades de entrenamiento.

## Entrenamiento

Para iniciar el entrenamiento del modelo HerdNet, ejecute el siguiente comando:

```bash
uv run python -m oficial_herdnet.scripts.train_herdnet
```

Este comando utilizará la configuración especificada en `train.yml` para entrenar el modelo.

## Evaluación y Predicción

[Instrucciones para evaluación y predicción pendientes]
