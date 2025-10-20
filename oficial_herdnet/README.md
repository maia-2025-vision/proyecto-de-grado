# HerdNet - Entrenamiento y Uso

## Instalación de Datos

Antes de comenzar el entrenamiento, es necesario obtener los datos utilizando DVC:

```bash
# Obtener los parches de imágenes para entrenamiento
uv run dvc pull patches-512-ol-160-mv01-train

# Obtener los parches de imágenes para validacion
uv run dvc pull patches-512-ol-160-mv01-val

# Obtener los parches de imágenes para test
uv run dvc pull patches-512-ol-160-mv01-test

```

```bash
# Para generar los parches desde los datos originales
# Si se requieren otras configuraciones hay que generar un nuevo stage en dvc.yaml
uv run dvc repro patches-512-ol-160-mv01-train

uv run dvc repro patches-512-ol-160-mv01-val

uv run dvc repro patches-512-ol-160-mv01-test

#cargar los nuevos datos al dvc remoto
uv run dvc push
```

## Configuración y Entrenamiento

Existen dos métodos para entrenar el modelo HerdNet:

### Método 1: Implementación oficial del autor (Hydra)

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
wandb_entity: 'maiavision2025-universidad-de-los-andes'  # Entidad en wandb
wandb_run: 'Train 1'  # Identificador de esta ejecución ¡Cambiar en cada ejecucion para no mezclar los datos!

# Si en entrenamiento implica cambio de los hiperparametros, es importante cambiar el work_dir para 
# configurar la salida de un nuevo modelo
training_settings:
  work_dir: 'data/models/herdnet_v1' 
```

#### Iniciar el entrenamiento oficial

Para iniciar el entrenamiento utilizando el script oficial con Hydra:  
**Si cambia la version crear otro stage en el `dvc.yaml`**

```bash
uv run dvc repro train-herdnet-v1  
```

Este comando utilizará la configuración especificada en `configs/train/herdnet.yaml`, y registrará el progreso del entrenamiento en wandb para su visualización y análisis.


### Método 2: Implementación personalizada (Simplificada)

La configuración del entrenamiento se encuentra en el archivo `oficial_herdnet/scripts/custom_train.yaml`:

En el `custom_train.yaml` modificar `work_dir: "data/models/herdnet_v1"` en caso de generar una versión diferente.

Para iniciar el entrenamiento con esta implementación simplificada:

```bash
uv run python -m oficial_herdnet.scripts.custom_train_herdnet
```

## Evaluación y Predicción

[Instrucciones para evaluación y predicción pendientes]
