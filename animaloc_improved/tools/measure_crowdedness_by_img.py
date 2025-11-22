import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from mistree import GetMST  # type: ignore[import-untyped]

from animaloc_improved.tools.latex_utils import df_to_latex

app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)

lbl_to_species = {
    1: "Alcelaphinae",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}


def _enrich_bbox_df(gt_df: pd.DataFrame) -> pd.DataFrame:
    gt_df = gt_df.copy()

    # subsets
    #   Virunga: Archivos con nombres de como S_##_##_##_DSC#######.JPG
    #   AED: Aerial Elephant Dataset = archivos con nombres de la forma <hexadecimal-largo>.JPG
    gt_df["subdataset"] = np.where(
        gt_df["images"].str.split("/").str[-1].str.contains("_"), "Virunga", "AED"
    )
    gt_df["order"] = gt_df["split"].map(lambda s: {"train": 0, "val": 1, "test": 2}[s])

    gt_df["A(bbox)"] = (  # en pixeles cuadrados
        (gt_df["x_max"] - gt_df["x_min"]) * (gt_df["y_max"] - gt_df["y_min"])
    )
    gt_df["L(bbox)"] = np.sqrt(gt_df["A(bbox)"])  # en pixeles (lineales)
    # calcular centros de las cajas
    gt_df["x"] = (gt_df["x_max"] + gt_df["x_min"]) * 0.5
    gt_df["y"] = (gt_df["y_max"] + gt_df["y_min"]) * 0.5

    return gt_df


def _report_cnt_by_split_subset(gt_df: pd.DataFrame) -> None:
    cnts_by_split_subset = (
        gt_df.groupby(["split", "subdataset"])
        .agg(
            {
                "labels": "count",
                "images": "nunique",
                "order": "mean",
            }
        )
        .reset_index()
        .sort_values("order")
    )

    total_labels = cnts_by_split_subset["labels"].sum()
    total_images = cnts_by_split_subset["images"].sum()
    cnts_by_split_subset["pct_labels"] = np.round(
        100 * (cnts_by_split_subset["labels"] / total_labels), 1
    )
    cnts_by_split_subset["pct_images"] = np.round(
        100 * (cnts_by_split_subset["images"] / total_images), 1
    )

    logger.info(f"Counts by split and subdataset:\n{cnts_by_split_subset.to_markdown()}")
    print(
        df_to_latex(
            cnts_by_split_subset[
                "split,subdataset,labels,pct_labels,images,pct_images".split(",")
            ].rename(
                columns={
                    "labels": r"$\#$ anots.",
                    "pct_labels": r"$\%$ anots.",
                    "images": r"$\#$ imágenes",
                    "pct_images": r"$\%$ imágenes",
                }
            ),
            caption=r"Conteo de imágenes y anotaciones por \textit{split} y \textit{subdataset}",
        )
    )


def _report_by_subds_lbl(gt_df: pd.DataFrame) -> None:
    cnts_by_subds_lbl = (
        pd.DataFrame(
            gt_df.groupby(["subdataset", "labels"]).agg(
                {
                    "x_min": "count",
                    "L(bbox)": "median",
                }
            )
        )
        .reset_index()
        .rename(columns={"x_min": "n_annots"})
    )
    cnts_by_subds_lbl["L(bbox)"] = np.round(cnts_by_subds_lbl["L(bbox)"], 1)

    total_annots = cnts_by_subds_lbl["n_annots"].sum()

    cnts_by_subds_lbl["Especie"] = cnts_by_subds_lbl["labels"].map(lambda lbl: lbl_to_species[lbl])
    cnts_by_subds_lbl["pct_annots"] = np.round(
        100 * cnts_by_subds_lbl["n_annots"] / total_annots, 1
    )

    logger.info(f"All cols: {list(cnts_by_subds_lbl.columns)}")
    cnts_by_subds_lbl = cnts_by_subds_lbl[
        "subdataset,Especie,n_annots,pct_annots,L(bbox)".split(",")
    ]

    logger.info(f"Label counts by subdataset and species:\n{cnts_by_subds_lbl.to_markdown()}")
    print(
        df_to_latex(
            cnts_by_subds_lbl.rename(
                columns={
                    "subdataset": "Subdataset",
                    "n_annots": "Bboxes\n{}$\\#$",
                    "pct_annots": "BBoxes.\n{}[\\%]",
                    "L(bbox)": "Mediana de L(bbox)\n{}[px]",
                }
            ),
            caption=r"Conteo de anotaciones por \textit{subdataset} y \textit{especie}",
        )
    )


def calc_median_mis_edge(img_df: pd.DataFrame) -> float:
    """Calcular la mediana de la longitud de las aristas del minimum spanning tree."""
    if len(img_df) == 1:
        return float("nan")  # solo una anotación, esto es lo menos crowded que se puede
    else:
        mis = GetMST(x=img_df["x"].values, y=img_df["y"].values)
        mis.get_stats(k_neighbours=len(img_df) - 1)
        return float(np.median(mis.edge_length))


def _counts_by_label(img_df: pd.DataFrame) -> dict[int, int]:
    counts_by_label_dict = img_df["labels"].value_counts().to_dict()
    labels_mode = img_df["labels"].mode()
    if len(labels_mode) > 1:
        print(f"labels_mode={labels_mode.values!r}")
    return pd.Series(  # type: ignore[return-value]
        {
            "dominant_species": labels_mode.iloc[0],  # resolve ties arbitrarily
            "counts_by_label": counts_by_label_dict,
            "has_mixed_species": len(counts_by_label_dict) > 1,
        }
    )


def _produce_stats_by_image(gt_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["split", "subdataset", "images", "rel_path"]
    image_meta = gt_df[meta_cols].drop_duplicates()  # una fila for imagen única
    logger.info(f"{len(image_meta)=}")

    num_annots_per_image = (
        gt_df.groupby("rel_path")
        .agg({"labels": "count", "L(bbox)": "median"})
        .reset_index()
        .rename(columns={"labels": "num_bboxes", "L(bbox)": "median_l_bbox"})
    )

    med_mis_edge_len = (
        gt_df.groupby("rel_path")
        .apply(calc_median_mis_edge, include_groups=False)  # type: ignore[call-overload]
        .reset_index()
        .rename(columns={0: "median_animal_sep"})
    )

    cnts_by_label = gt_df.groupby("rel_path").apply(
        _counts_by_label,
        include_groups=False,  # type: ignore[call-overload]
    )
    print(cnts_by_label, type(cnts_by_label))

    stats_by_image = (
        image_meta.merge(num_annots_per_image, on="rel_path")
        .merge(cnts_by_label, on="rel_path")
        .merge(med_mis_edge_len, on="rel_path")
    )

    stats_by_image["counts_by_label"] = stats_by_image["counts_by_label"].apply(json.dumps)

    stats_by_image["normalized_animal_sep"] = (
        stats_by_image["median_animal_sep"] / stats_by_image["median_l_bbox"]
    )
    stats_by_image = stats_by_image.sort_values("normalized_animal_sep")

    return stats_by_image


@app.command()
def main(
    gt_base_dir: Path = typer.Option(
        Path("data/groundtruth/csv"), "--gt-dir", help="Path to ground truth csv directory"
    ),
    input_suffix: str = typer.Option(
        "_big_size_A_B_E_K_WH_WB-fixed-header.csv",
        help="suffix for bbox annotation files to be taken into account",
    ),
    out_dir: Path = typer.Option(..., help="output directory"),
) -> None:
    # gt_fnames = ["train_big_size_A_B_E_K_WH_WB-fixed-header.csv",
    #     "val_big_size_A_B_E_K_WH_WB-fixed-header.csv",
    #     "test_big_size_A_B_E_K_WH_WB-fixed-header.csv"]
    gt_paths = list(gt_base_dir.glob(f"*{input_suffix}"))
    logger.info(f"Loading ground truth annotations from {gt_paths!s}")
    gt_parts = []
    for gt_path in gt_paths:
        df = pd.read_csv(gt_path)
        df["split"] = gt_path.name.split("_")[0]
        df["rel_path"] = df["split"] + "/" + df["images"]
        gt_parts.append(df)

    gt_df = pd.concat(gt_parts)

    gt_df = _enrich_bbox_df(gt_df)

    cnts_by_split = gt_df.groupby("split").count()[["labels"]].reset_index()
    logger.info(f"Counts by split:\n{cnts_by_split.to_markdown()}")

    # cnts_by_split_subset = gt_df.groupby(["split", "subdataset"])
    # .count()[['labels']].reset_index()
    _report_cnt_by_split_subset(gt_df)
    _report_by_subds_lbl(gt_df)

    stats_by_image = _produce_stats_by_image(gt_df)

    out_dir.mkdir(exist_ok=True, parents=True)
    stats_by_img_path = out_dir / input_suffix.strip("_")
    logger.info(f"\n{stats_by_image.sample(50).to_markdown()}")

    logger.info(f"Writing stats by image to {stats_by_img_path!s}")
    stats_by_image.to_csv(stats_by_img_path, index=False)


if __name__:
    app()
