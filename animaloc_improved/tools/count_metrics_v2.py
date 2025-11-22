"""Example invocation

python animaloc_improved/tools/count_metrics_v2.py \
   --gt "data/gt-preprocessed/csv/test_big_size_A_B_E_K_WH_WB-points.csv" \
   --preds "data/test_results/herdnet_v2_full_imgs/detections.csv" \
   --out_dir ./
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from animaloc_improved.tools.csv_tool import flatten_df_column_names

app = typer.Typer()

OUTPUT_DECIMALS = 1


def _round_means_ip(aggregated: pd.DataFrame) -> None:
    for col in aggregated.columns:
        if col.startswith("M") or col == "RMSE":
            aggregated[col] = aggregated[col].round(OUTPUT_DECIMALS)


def _load_and_agg_g_truth(g_truth_path: Path) -> pd.DataFrame:
    g_truth = pd.read_csv(g_truth_path)

    logger.info(
        f"Loaded ground truth data {g_truth.shape=}, unique images: {g_truth['images'].nunique()=}"
    )

    # pick a column for counting
    count_col = "x" if "x" in g_truth.columns else "x_min"

    # aggregate by image and label
    g_truth_cnts = (
        g_truth.groupby(["images", "labels"])
        .agg({count_col: "count"})
        .rename(columns={count_col: "true_count"})
        .reset_index()
    )

    return g_truth_cnts


def _load_and_agg_preds(preds_path: Path) -> pd.DataFrame:
    predicted = pd.read_csv(preds_path)
    logger.info(
        f"Loaded prediction data {predicted.shape=}, unique images {predicted['images'].nunique()=}"
    )
    invalid_row_mask = predicted["labels"].isnull()
    logger.info(
        f"{sum(invalid_row_mask)} invalid rows found, out of {len(predicted)}, dropping them"
    )
    predicted = predicted.loc[~invalid_row_mask].copy()
    predicted["labels"] = predicted["labels"].astype(int)

    count_col = "x" if "x" in predicted.columns else "x_min"
    predicted_cnts = (
        predicted.groupby(["images", "labels"])
        .agg({count_col: "count"})
        .rename(columns={count_col: "pred_count"})
        .reset_index()
    )

    return predicted_cnts


def merge_and_add_errors(g_truth_cnts: pd.DataFrame, predicted_cnts: pd.DataFrame) -> pd.DataFrame:
    merged = g_truth_cnts.merge(predicted_cnts, on=["images", "labels"], how="outer")

    merged["true_count"] = merged["true_count"].fillna(0.0)
    merged["pred_count"] = merged["pred_count"].fillna(0.0)

    # Plain counting error, can be positive or negative
    merged["count_err"] = merged["pred_count"] - merged["true_count"]

    # Absolute error : abs(count_error)
    merged["abs_err"] = np.abs(merged["count_err"])

    # Squared error
    merged["sq_err"] = (merged["pred_count"] - merged["true_count"]) ** 2

    # Count error as a percent of true count, only defined when true count > 0
    merged["pct_err"] = np.where(
        merged["true_count"] > 0,
        100 * (merged["pred_count"] - merged["true_count"]) / merged["true_count"],
        np.nan,
    )
    merged["abs_pct_err"] = np.abs(merged["pct_err"])

    return merged


def _rename_agg_cols(aggregated: pd.DataFrame) -> pd.DataFrame:
    aggregated = flatten_df_column_names(aggregated)
    aggregated = aggregated.rename(
        columns={
            "true_count_mean": "M(C)",
            "true_count_sum": "Σ(C)",
            "pred_count_mean": "M(Ĉ)",
            "pred_count_sum": "Σ(Ĉ)",
            "count_err_mean": "ME",  # mean error, without taking abs value can be positive or neg
            "abs_err_mean": "MAE",  # mean absolute error
            "sq_err_mean": "MSE",  # mean squared error
            "pct_err_mean": "MPE",  # mean percent error
            "abs_pct_err_mean": "MAPE",
        }
    )

    return aggregated


def _calc_means_by_label(merged: pd.DataFrame) -> pd.DataFrame:
    means_by_label = (
        merged.groupby("labels")
        .agg(
            {
                "true_count": ("mean", "sum"),  # type: ignore[dict-item]
                "pred_count": ("mean", "sum"),  # type: ignore[dict-item]
                "count_err": "mean",
                "abs_err": "mean",
                "sq_err": "mean",
                "pct_err": "mean",
                "abs_pct_err": "mean",
            }
        )
        .reset_index()
    )
    means_by_label = flatten_df_column_names(means_by_label)
    means_by_label = _rename_agg_cols(means_by_label)
    means_by_label["RMSE"] = means_by_label["MSE"] ** 0.5
    _round_means_ip(means_by_label)

    return means_by_label


def agg_by_image_and_add_errors(merged: pd.DataFrame) -> pd.DataFrame:
    bin_metrics = (
        merged.groupby("images")
        .agg(
            {
                "true_count": "sum",
                "pred_count": "sum",
                "count_err": "sum",
            }
        )
        .reset_index()
    )
    bin_metrics["abs_err"] = np.abs(bin_metrics["count_err"])
    bin_metrics["sq_err"] = bin_metrics["count_err"] ** 2
    bin_metrics["pct_err"] = np.where(
        bin_metrics["true_count"] > 0,
        (bin_metrics["count_err"] / bin_metrics["true_count"]) * 100,
        np.nan,
    )
    bin_metrics["abs_pct_err"] = np.abs(bin_metrics["pct_err"])
    return bin_metrics


def _calc_binary_means(bin_metrics: pd.DataFrame) -> pd.DataFrame:
    bin_metrics["Especie"] = "binaria"
    bin_metrics_agg = (
        bin_metrics.groupby("Especie")
        .agg(
            {
                "true_count": ("mean", "sum"),  # type: ignore[dict-item]
                "pred_count": ("mean", "sum"),  # type: ignore[dict-item]
                "count_err": "mean",
                "abs_err": "mean",
                "sq_err": "mean",
                "pct_err": "mean",
                "abs_pct_err": "mean",
            }
        )
        .reset_index()
    )
    bin_metrics_agg = _rename_agg_cols(bin_metrics_agg)
    bin_metrics_agg["RMSE"] = np.sqrt(bin_metrics_agg["MSE"])
    _round_means_ip(bin_metrics_agg)

    return _rename_agg_cols(bin_metrics_agg)


@app.command()
def main(
    g_truth_path: Path = typer.Option(
        ..., "--ground-truth", "--gt", help="Path to the ground truth csv file"
    ),
    preds_path: Path = typer.Option(..., "-p", "--preds", help="Path to the predictions csv file"),
    idx_2_species_path: Path = typer.Option(
        Path("data/idx_2_species.json"), "--i2s", help="Path to the index-2-species json file"
    ),
    out_dir: Path = typer.Option(..., "-o", "--out-dir", help="Path to the output directory"),
) -> None:
    g_truth_cnts = _load_and_agg_g_truth(g_truth_path)
    predicted_cnts = _load_and_agg_preds(preds_path)

    idx_2_species = json.loads(idx_2_species_path.read_text())

    merged = merge_and_add_errors(g_truth_cnts, predicted_cnts)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path1 = out_dir / "g_truth_pred_cnts_errors_by_image_species.csv"
    logger.info(
        f"Writing merged data frame with gt-counts, pred-counts and errors by "
        f"[Image, Especie] to: {out_path1}"
    )
    merged.to_csv(out_path1, index=False)

    means_by_label = _calc_means_by_label(merged)

    means_by_label["Especie"] = (
        means_by_label["labels"].astype(str).map(lambda lbl: idx_2_species[lbl])
    )
    means_by_label = means_by_label.drop("labels", axis=1)
    sorted_cols = ["Especie"] + [col for col in means_by_label if col != "Especie"]
    means_by_label = means_by_label[sorted_cols]

    bin_metrics = agg_by_image_and_add_errors(merged)
    out_path2 = out_dir / "g_truth_pred_cnts_errors_by_image.csv"
    logger.info(
        f"Writing merged data frame with gt-counts, pred-counts and errors by [Image] to:"
        f"{out_path2}"
    )
    bin_metrics.to_csv(out_path2, index=False)

    means_binary = _calc_binary_means(bin_metrics)

    all_metrics = pd.concat([means_by_label, means_binary]).drop(columns=["MSE"])
    selected_metrics = all_metrics[
        ["Especie", "Σ(C)", "M(C)", "M(Ĉ)", "MAE", "RMSE", "MPE", "MAPE"]
    ]
    logger.info(f"Aggregated count metrics:\n{selected_metrics.to_markdown()}")
    out_path3 = out_dir / "aggregated_count_metrics_v2.csv"
    all_metrics.to_csv(out_path3, index=False)

    logger.info(
        "Outputs written:\n" + "\n".join([f"- {p!s}" for p in [out_path1, out_path2, out_path3]])
    )


if __name__ == "__main__":
    app()
