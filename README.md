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

## Papers que tal vez podríamos revisar después si tenemos tiempo...

1. [WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification](https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.pdf)
   - Código aquí: https://github.com/UCSB-VRL/WildlifeMapper
2. [Naidu et. al. 2025 - DEAL-YOLO: DRONE-BASED EFFICIENT ANIMAL LOCALIZATION USING YOLO](https://arxiv.org/pdf/2503.04698)
2. [Wang, Gao 2025 - SF-DETR: A Scale-Frequency Detection Transformer for Drone-View Object Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991380/) (No parece tener código asociado)
3. [Zhu, Zhang 2025 - Efficient vision transformers with edge enhancement for robust small target detection in drone-based remote sensing](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1599099/full)
  (No parece tener código asociado)
4. [Jie Hu et. al. 2025 - A small object detection model for drone images based on multi-attention fusion network](https://www.sciencedirect.com/science/article/abs/pii/S0262885625000241)
   (No parece tener código asociado)
