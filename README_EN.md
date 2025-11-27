<div id="top"></div>

<div align="center">
  <h1>Proyecto de Grado MAIA 2025 — Animal Detection &amp; Counting</h1>
  <p>
    Deep-learning pipeline for detection and counting of herd animals from aerial imagery.<br/>
    HerdNet + Faster R-CNN models, FastAPI inference service, Streamlit dashboard.
  </p>
  <p>
    <a href="#getting-started">Get Started</a> ·
    <a href="#usage">API Usage</a> ·
    <a href="#dashboard-guide">Dashboard Guide</a> ·
    <a href="#roadmap">Roadmap</a>
  </p>
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.13+-blue">
    <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688">
    <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b">
    <img alt="DVC" src="https://img.shields.io/badge/Data-DVC-945DD6">
    <img alt="Status" src="https://img.shields.io/badge/Status-WIP-orange">
  </p>
</div>

## Table of Contents
1. [About](#about)
2. [Built With](#built-with)
3. [Developer Guide](#developer-guide)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Dashboard Guide](#dashboard-guide)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)
11. [Acknowledgments](#acknowledgments)

## About
- Goal: Detect and count multiple African mammal species (Virunga + AED datasets) in dense herds from aerial imagery.
- Models: HerdNet (density/points) and Faster R-CNN (boxes) with multiple training variants (e.g., HN-4, FRC-50/101).
- Deliverables: reproducible pipelines (DVC), experiment tracking (W&B), FastAPI inference service, and Streamlit dashboard for non-technical users.
[↑ back to top](#top)

## Built With
- Python, PyTorch, Torchvision
- DVC for data/model versioning
- Hydra configs (HerdNet), Pydantic/FastAPI for API
- Streamlit for dashboard UI
- Docker for API/UI packaging
[↑ back to top](#top)

## Developer Guide
### Dev setup (uv)
```bash
uv python install          # uses .python-version
uv sync --all-extras       # install deps
source .venv/bin/activate
```

### Data & models (DVC)
- Minimum for inference (HN):  
  ```bash
  dvc pull data/models/herdnet_v2/best_model.pth
  dvc pull data/groundtruth data/train data/val data/test
  ```
- Optional: fetch patches/results as needed per stage in `dvc.yaml` (e.g., `dvc pull data/patches-512-ol-160-m01`).

### Run backend
- Local: `poe serve-hn` (HerdNet) or `poe serve-frc` (FRC) — API on `http://localhost:8000`.
- Docker: `poe dockerize-api` then `poe docker-run-api-hnet-prof` (mounts AWS creds if pulling from S3).

### Repo layout (1-line map)
- `api/` FastAPI service (entry: `api/main.py`, deps config in `pyproject.toml`).
- `dashboard/` Streamlit UI (`dashboard/app.py`, `pages/`).
- `oficial_herdnet/` Training scripts/configs (Hydra) for HerdNet.
- `configs/` Model/test configs (API and training).
- `data/` DVC-managed datasets, groundtruths, models.
- `dvc.yaml` Pipelines (train/test/infer); `dvc.lock` pinned versions.
- `pyproject.toml` Tooling, tasks (`poe`), deps.
- `notebooks/`, `marimo-nbs/` Exploratory analysis and experiments.
- `animaloc_improved/` Utilities for inference/hard-negatives.

### Minimal training commands
- Example HerdNet train/test:  
  ```bash
  uv run dvc repro train-herdnet-v4
  uv run dvc repro test-herdnet-v4-full-imgs
  ```
- Fetch required inputs beforehand (patches/groundtruth) per `dvc.yaml` stages if not present.
[↑ back to top](#top)

## Getting Started
### Prerequisites
- `astral-uv` installed (see https://docs.astral.sh/uv/getting-started/installation/)
- Python version from `.python-version`
- DVC configured (`dvc pull` requires access to S3 remote)

### Setup
```bash
uv python install
uv sync --all-extras
source .venv/bin/activate
dvc pull  # fetch data/models
```

### Run services
```bash
# API (choose one)
poe serve-hn   # HerdNet
poe serve-frc  # Faster-RCNN

# Dashboard
poe start-dashboard

# Docker builds
poe dockerize-api
poe dockerize-dashboard
```

Model/config paths for API are set in `pyproject.toml` under `[tool.poe.tasks.serve-*]` (`MODEL_WEIGHTS_PATH`, `MODEL_CFG_PATH`).
[↑ back to top](#top)

## Usage
### API (FastAPI)
- Docs UI: http://localhost:8000/docs
- Main endpoint: `POST /predict` accepts an image; returns detections/centroids per species.
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```
[↑ back to top](#top)

### Dashboard Guide
- Start: `poe start-dashboard` (API at `localhost:8000`). Docker: `poe dockerize-dashboard` -> `poe docker-run-dashboard` (http://localhost:8501).
- Steps (with suggested screenshots under `docs/img/`):
  1) Welcome — navigate via sidebar. (`dashboard-step1-welcome.png`)
  2) Upload & process — drag/drop `.jpg/.png/.jpeg`; set Label (e.g., `Kruger_Sur_2025-11-10`) and capture date/time; click “Procesar Imágenes”. (`dashboard-step2-carga.png`)
  3) Monitor — spinner while uploading to S3 and calling API; success/error message. (`dashboard-step3-progreso.png`)
  4) Load results — in “Visualizador y Métricas de Detección”, pick Región (Label) and Sobrevuelo (date/time), click “Cargar Resultados”. (`dashboard-step4-selectores.png`)
  5) Metrics — table of species counts + bar chart. (`dashboard-step5-metricas.png`)
  6) Detections — choose an image; view boxes and/or centroids overlay. (`dashboard-step6-detecciones.png`)
  7) Controls — sidebar: display mode (boxes/centroids/both), confidence threshold, line/point size, species filter. (`dashboard-step7-controles.png`)
[↑ back to top](#top)

## Roadmap
- [ ] Add sample images + filled screenshot assets under `docs/img/`.
- [ ] Document model zoo (HN-1..4, FRC-50/101) with metrics and download links.
- [ ] Add deployment recipes (API + dashboard) for cloud.
- [ ] Expand tests (API e2e, dashboard smoke).
[↑ back to top](#top)

## Contributing
- Install pre-commit hooks: `pre-commit install`
- Lint/type-check: `poe check`, `poe type-check`
- e2e tests:
  ```bash
  poe e2e predict-one
  poe e2e predict-many
  poe e2e predict-one-mult
  ```
- Open PRs against main; keep DVC-tracked artifacts out unless necessary.
[↑ back to top](#top)

## License

[↑ back to top](#top)

## Contact
- Andrés Alea — a.alea@uniandes.edu.co
- Mixer Gutiérrez — mf.gutierreza1@uniandes.edu.co
- Jose Daniel Pineda — jd.pineda@uniandes.edu.co
- Mateo Restrepo — m.restrepom2@uniandes.edu.co
[↑ back to top](#top)

## Acknowledgments
The authors would like to thank teacher Isaí Daniel Chacón from Universidad de los Andes for the invaluable advice and guidance provided throughout this project.
[↑ back to top](#top)

## References
- HerdNet repo by Alexandre Delplanque (training pipeline base)
- Datasets: Virunga + Aerial Elephant Dataset
- Tools: DVC, W&B, Streamlit, FastAPI
[↑ back to top](#top)
