# HerdNet - Entrenamiento y Uso

## Instalación de Datos

Antes de comenzar el entrenamiento, es necesario obtener los datos utilizando DVC:

```bash
# Obtener los parches de imágenes para entrenamiento
uv run dvc pull patches_512.dvc

# Obtener los modelos pre-entrenados (si se requiere)
uv run dvc pull models.dvc
```

## Configuración y Entrenamiento

Existen dos métodos para entrenar el modelo HerdNet:

### Método 1: Implementación personalizada (Simplificada)

La configuración del entrenamiento se encuentra en el archivo `oficial_herdnet/scripts/custom_train.yml`:

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

Para iniciar el entrenamiento con esta implementación simplificada:

```bash
uv run python -m oficial_herdnet.scripts.custom_train_herdnet
```

### Método 2: Implementación oficial del autor (Hydra)

La implementación original del autor utiliza Hydra para la gestión de configuraciones. El archivo de configuración principal se encuentra en `oficial_herdnet/configs/train/herdnet.yaml`.

#### Configuración de Weights & Biases (wandb)

La implementación oficial utiliza wandb para registrar el progreso del entrenamiento. Antes de ejecutar el entrenamiento, es necesario configurar wandb:

1. Crear una cuenta en [wandb.ai](https://wandb.ai) si aún no la tienes
2. Iniciar sesión en la terminal con:

```bash
wandb login
```

3. Editar los siguientes parámetros en `oficial_herdnet/configs/train/herdnet.yaml`:

```yaml
wandb_project: 'Herdnet'  # Nombre del proyecto en wandb
wandb_entity: 'tu-nombre-de-usuario'  # Tu nombre de usuario o equipo en wandb
wandb_run: 'Train 1'  # Identificador de esta ejecución
```

#### Iniciar el entrenamiento oficial

Para iniciar el entrenamiento utilizando el script oficial con Hydra:

```bash
uv run python -m oficial_herdnet.tools.train
```

Este comando utilizará la configuración especificada en `configs/train/herdnet.yaml`, y registrará el progreso del entrenamiento en wandb para su visualización y análisis.

## Evaluación y Predicción

[Instrucciones para evaluación y predicción pendientes]
