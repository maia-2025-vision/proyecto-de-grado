# HerdNet - Entrenamiento y Uso

## Instalación de Datos

Antes de comenzar el entrenamiento, es necesario obtener los datos utilizando DVC:

```bash
# Obtener los parches de imágenes
uv run dvc pull patches-512-ol-160-mv01-train
uv run dvc pull patches-512-ol-160-mv01-val
uv run dvc pull patches-512-ol-160-mv01-test
```

### Generación de parches desde datos originales

```bash
# Para generar los parches desde los datos originales
uv run dvc repro patches-512-ol-160-mv01-train
uv run dvc repro patches-512-ol-160-mv01-val
uv run dvc repro patches-512-ol-160-mv01-test

# Cargar los nuevos datos al DVC remoto
uv run dvc push
```

> **Nota importante**: Si se requieren otras configuraciones, se debe generar un nuevo stage en `dvc.yaml`

## Configuración y Entrenamiento

### Método 1: Implementación oficial (Hydra)

#### Configuración de Weights & Biases

1. Crear cuenta en [wandb.ai](https://wandb.ai)
2. Iniciar sesión: `wandb login`
3. Editar parámetros en `oficial_herdnet/configs/train/herdnet.yaml`:

```yaml
wandb_project: 'Herdnet'
wandb_entity: 'maiavision2025-universidad-de-los-andes'
wandb_run: 'Train 1'

training_settings:
  work_dir: 'data/models/herdnet_v1'
```

#### Ejecutar entrenamiento

```bash
uv run dvc repro train-herdnet-v1
```

### Método 2: Implementación personalizada

Configurar `work_dir` en `oficial_herdnet/scripts/custom_train.yaml` y ejecutar:

```bash
uv run python -m oficial_herdnet.scripts.custom_train_herdnet
```

## Evaluación y Testing

```bash
uv run dvc repro test-herdnet-v1
```

> **Nota**: El stitcher se desactiva automáticamente para el conjunto de test con parches fijos de 512x512.

## Inferencia

Colocar imágenes en `outputs/infer/herdnet_v1/` y ejecutar:

```bash
uv run python oficial_herdnet/tools/infer.py outputs/infer/herdnet_v1/ data/models/herdnet_v1/best_model.pth -size 512 -over 160
```
