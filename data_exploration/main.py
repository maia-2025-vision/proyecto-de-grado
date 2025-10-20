import pandas as pd

from utils import (
    analyze_area_statistics,
    analyze_dataset,
    generate_split_table,
    get_top_n_bboxes_in_area_range,
    load_all_bbox_data,
    plot_bbox_area_histogram_from_data,
    plot_total_species_comparison,
)


def generate_markdown_report(
    dataset_summaries,
    output_md_path,
    class_distribution_image_name,
    stats_absolute,
    stats_relative_global,
    stats_relative_split,
    image_sizes,
    top_bboxes=None,
):
    """Genera un solo archivo Markdown con toda la información combinada."""
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Exploración de Datos\n\n")

        for name, summary in dataset_summaries.items():
            f.write(f"## Dataset: `{name}`\n\n")
            for split in ["train", "val", "test"]:
                f.write(f"### Split: `{split}`\n\n")
                f.write(generate_split_table(summary.get(split, {})))
                f.write("\n\n")

        f.write("## Comparación Total por Especie\n\n")
        f.write(f"![Gráfica comparativa]({class_distribution_image_name})\n\n")

        f.write("## Distribución de Áreas de Bounding Boxes por Clase\n\n")
        for split in ['train', 'val', 'test']:
            f.write(f"### Histograma de Áreas - Split `{split}`\n\n")
            f.write(f"![Histograma áreas {split}](bbox_area_{split}.png)\n\n")

        f.write("## Estadísticas Descriptivas de Áreas\n\n")
        f.write("### Estadísticas de Áreas Absolutas (px²)\n\n")
        f.write(stats_absolute.round(2).to_markdown())
        f.write("\n\n")
        f.write("### Estadísticas de Áreas Relativas (% de imagen)\n\n")
        f.write(stats_relative_global.round(4).to_markdown())
        f.write("\n\n")

        f.write("### Estadísticas de Áreas Relativas por Split (% de imagen)\n\n")
        for split in ["train", "val", "test"]:
            subset = stats_relative_split.xs(split, level="split", drop_level=False)
            if not subset.empty:
                f.write(f"#### Split: `{split}`\n\n")
                f.write(subset.round(4).to_markdown())
                f.write("\n\n")

        f.write("### Distribución de Tamaños de Imagen por Split\n\n")
        for split in ["train", "test", "val"]:
            subset = image_sizes[image_sizes["split"] == split]
            if not subset.empty:
                f.write(f"#### Split: `{split}`\n\n")
                subset_sorted = subset.sort_values("count", ascending=False)
                f.write(subset_sorted.to_markdown(index=False))
                f.write("\n\n")

        if top_bboxes:
            f.write("## Top BBoxes en Rango Específico de Área\n\n")
            df = pd.DataFrame(top_bboxes)
            # Puedes elegir mostrar solo algunas columnas si lo deseas:
            columns_to_show = [
                "dataset",
                "split",
                "category",
                "area",
                "area_relative",
                "image_name",
                "bbox",
            ]
            df = df[columns_to_show]
            f.write(df.to_markdown(index=False))
            f.write("\n\n")


def main():
    datasets = {
        "big_size": "data/groundtruth/json/big_size",
        "subframes": "data/groundtruth/json/sub_frames",
    }

    # Analizar ambos datasets
    dataset_summaries = {name: analyze_dataset(path, name) for name, path in datasets.items()}

    # Gráfico comparativo
    class_distribution_image_name = "class_distribution.png"
    image_path = f"data_exploration/result/{class_distribution_image_name}"
    plot_total_species_comparison(dataset_summaries, image_path)

    # bbox areas para conjunto de entrenamiento
    bbox_datasets = {"subframes": "data/groundtruth/json/sub_frames"}
    bbox_data = load_all_bbox_data(bbox_datasets)

    plot_bbox_area_histogram_from_data(bbox_data, "data_exploration/result/")

    stats_absolute, stats_relative_global, stats_relative_split, image_sizes = (
        analyze_area_statistics(bbox_data)
    )

    top_bboxes = get_top_n_bboxes_in_area_range(bbox_data, 1000, 13000, n=5)

    # Generar markdown
    md_path = "data_exploration/result/exploracion_datos.md"

    generate_markdown_report(
        dataset_summaries,
        md_path,
        class_distribution_image_name,
        stats_absolute,
        stats_relative_global,
        stats_relative_split,
        image_sizes,
        top_bboxes,
    )

    print(f"\nInforme generado en: {md_path}")


if __name__ == "__main__":
    main()
