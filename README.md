# Proyecto de Grado MAIA 2025
Desarrollo de un modelo de deep learning para el conteo y detección de animales en manadas densas a partir de imágenes aéreas

# Setup inicial

Lo siguiente asume que `astral-uv` está ya instalado a nivel de sistema operativo.
Para instrucciones de instalación referirse a:
https://docs.astral.sh/uv/getting-started/installation/

```bash
# instala la versión de python especificada en .python-version
uv python install

# instala las dependencias del grupo [dev] definidas en pyproject.toml
uv sync --all-extras
source .venv/bin/activate
```

## Activar checks de código automáticos antes de commit

Altamente recomendado!

```bash
pre-commit install
```

### Obtener datos de repo remoto DVC remoto

```bash
dvc pull  # .venv debe estar activado
```

### Arrancar api en local 


```bash
poe serve
```

Para cambiar el modelo usado por la API cambiar el valor de MODEL_PATH en la sección `[poe.tasks.serve]` 
de pyproject.toml. 


### Enlaces para miembros del equipo

1. [Carpeta compartida en google drive](https://drive.google.com/drive/folders/1zJC_QlJhYr01Lml5BTW8EZx73Uyl1Myi?usp=drive_link)
2. [Carpeta con Artículos bajados en PDF](https://drive.google.com/drive/folders/1JAqXpCSRE6jkqYFGxdxsxtr8hDjmOQ8F?usp=drive_link)

