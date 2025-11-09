import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_dataset(base_path, split_suffix):
    """Analiza los splits (train/val/test) de un dataset y retorna estadísticas detalladas."""
    splits = ["train", "val", "test"]
    summary = {}

    for split in splits:
        file_path = os.path.join(base_path, f"{split}_{split_suffix}_A_B_E_K_WH_WB.json")
        with open(file_path) as f:
            data = json.load(f)

        num_images = len(data["images"])
        num_annotations = len(data["annotations"])

        category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
        class_counts = Counter([ann["category_id"] for ann in data["annotations"]])
        class_counts_named = {category_map[k]: v for k, v in class_counts.items()}

        sizes = {f"{img['width']}x{img['height']}" for img in data["images"]}

        summary[split] = {
            "num_images": num_images,
            "num_annotations": num_annotations,
            "class_distribution": class_counts_named,
            "image_sizes": sorted(sizes),
        }

    return summary


def get_total_species_counts(summary):
    """Suma los conteos de anotaciones por especie en todos los splits."""
    total_counts = Counter()
    for split_info in summary.values():
        total_counts.update(split_info["class_distribution"])
    return dict(total_counts)


def plot_total_species_comparison(dataset_summaries, output_path):
    """Genera una gráfica comparativa de totales por especie para varios datasets."""
    all_species = sorted(
        {
            species
            for summary in dataset_summaries.values()
            for species in get_total_species_counts(summary).keys()
        }
    )

    df = pd.DataFrame(
        {
            name: {
                species: get_total_species_counts(summary).get(species, 0)
                for species in all_species
            }
            for name, summary in dataset_summaries.items()
        }
    )

    ax = df.plot(kind="bar", figsize=(12, 6))
    plt.title("Distribución total de anotaciones por especie")
    plt.ylabel("Número de anotaciones")
    plt.xlabel("Especie")
    plt.xticks(rotation=45, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, padding=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return df


def load_all_bbox_data(datasets, splits: list[str] | None = None):
    """Carga info detallada de todas las bbox, incluye path completo a la imagen."""
    splits = splits or ["train", "val", "test"]

    all_bboxes = []
    for dataset_name, base_path in datasets.items():
        for split in splits:
            json_path = os.path.join(base_path, f"{split}_{dataset_name}_A_B_E_K_WH_WB.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                data = json.load(f)

            category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
            images_map = {img["id"]: img for img in data["images"]}

            for ann in data["annotations"]:
                bbox = ann["bbox"]
                _, _, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

                img_info = images_map[ann["image_id"]]
                area = w * h
                image_area = img_info["width"] * img_info["height"]
                area_relative = (area / image_area) * 100  # Porcentaje

                all_bboxes.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "image_id": ann["image_id"],
                        "image_name": img_info["file_name"],
                        "category": category_map[ann["category_id"]],
                        "bbox": bbox,
                        "area": area,
                        "area_relative": area_relative,
                        "image_width": img_info["width"],
                        "image_height": img_info["height"],
                    }
                )

    return all_bboxes


def get_top_n_bboxes_in_area_range(bbox_data, area_min, area_max, n=5):
    """Retorna top n bboxes cuya área está entre area_min y area_max.

    Las cajas se ordenan de manera descendentemente por área.
    """
    filtered = [b for b in bbox_data if area_min <= b["area"] <= area_max]
    filtered.sort(key=lambda x: x["area"], reverse=True)
    return filtered[:n]


def plot_bbox_area_histogram_from_data(bbox_data, output_dir):
    """Genera un histograma de áreas relativas por clase.

    Se separa cada clase por split, con colores distintos.
    """
    df = pd.DataFrame(bbox_data)
    splits = df["split"].unique()

    # Colores personalizados por split
    split_colors = {
        "train": "lightcoral",
        "val": "deepskyblue",
        "test": "goldenrod",
    }

    for split in splits:
        df_split = df[df["split"] == split]
        all_areas = {}
        for entry in df_split.to_dict(orient="records"):
            all_areas.setdefault(entry["category"], []).append(entry["area_relative"])

        sorted_classes = sorted(all_areas.keys())
        n_classes = len(sorted_classes)
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        color = split_colors.get(split, "gray")

        for idx, class_name in enumerate(sorted_classes):
            ax = axes[idx]
            ax.hist(all_areas[class_name], bins=30, color=color, edgecolor="black")
            ax.set_title(f"{class_name}")
            ax.set_xlabel("Área Relativa (% de la imagen)")
            ax.set_ylabel("Frecuencia")
            ax.tick_params(axis="x", labelrotation=45)

        # Eliminar subplots vacíos
        for j in range(len(sorted_classes), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Distribución de áreas relativas - Split: {split}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(output_dir, f"bbox_area_{split}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()


def generate_split_table(split_info):
    """Genera una tabla Markdown para un split específico."""
    lines = []
    lines.append(f"- **Número de imágenes**: {split_info['num_images']}")
    lines.append(f"- **Número de anotaciones**: {split_info['num_annotations']}")
    lines.append("- **Tamaños únicos de imagen**:")
    for size in split_info["image_sizes"]:
        lines.append(f"  - {size}")

    # Tabla con distribución por especie
    dist = split_info["class_distribution"]
    if dist:
        df = pd.DataFrame.from_dict(dist, orient="index", columns=["Anotaciones"])
        df.index.name = "Especie"
        lines.append("\n**Distribución por especie:**\n")
        lines.append(df.to_markdown())
    else:
        lines.append("\n(No hay anotaciones en este split)\n")
    return "\n".join(lines)


def analyze_area_statistics(bbox_data):
    """Analiza estadísticas descriptivas de las áreas por clase y por split."""
    df = pd.DataFrame(bbox_data)

    # Estadísticas por categoría y split (relativas)
    stats_relative_split = df.groupby(["split", "category"])["area_relative"].describe()

    # También puedes obtener estadísticas globales si quieres mantenerlas
    stats_absolute = df.groupby("category")["area"].describe()
    stats_relative_global = df.groupby("category")["area_relative"].describe()

    image_sizes = (
        df.groupby(["split", "image_width", "image_height"]).size().reset_index(name="count")
    )

    return stats_absolute, stats_relative_global, stats_relative_split, image_sizes


def summarize_points_gt(fp: Path):
    print(f"archivo: {fp!s}")
    df = pd.read_csv(fp)
    cols = list(df.columns)
    print(f"\n{len(df)} filas - {len(cols)} columnas: [{', '.join(cols)}]")

    print("\n## Número de valores únicos por columna:")
    print(pd.Series({col: df[col].nunique() for col in cols}).to_markdown())

    print("\n## Conteos de nulos por columna:")
    print(df.isnull().sum().reset_index().to_markdown())

    print("\n## Estadísticas de coordenadas de puntos (x, y)")
    print(df[["x", "y"]].describe().to_markdown())

    print("\n Frecuencias de labels:")
    freqs = df[["labels"]].value_counts().reset_index()
    freqs["pct"] = freqs["count"] / freqs["count"].sum() * 100
    print(freqs.sort_values("labels").to_markdown())
