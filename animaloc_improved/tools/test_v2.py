import json
from pathlib import Path

import pandas as pd
import torch
import typer
from loguru import logger

import animaloc_improved.tools.infer_metrics as im
from animaloc_improved.tools.common import SPECIES_MAP, NumpyEncoder

DEFAULT_MODEL_PATH = Path("data/models/herdnet_v2_hn2/best_model.pth")
DEFAULT_GT_PATH = Path("data/gt-preprocessed/csv/test_big_size_A_B_E_K_WH_WB-fixed-header.csv")
DEFAULT_IMG_ROOT = Path("data/test/")
DEFAULT_MATCH_TOL_BY_TRAT = {
    "point2point": 5,  # pixels of distance between pred and closest gt point
    "point2bbox": 0.5,  # fraction of width and height of bbox
}

app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)


def get_single_image_gt(gt_df: pd.DataFrame, image_name: str) -> pd.DataFrame:
    """Extract ground truth data for a specific image."""
    image_data = gt_df[gt_df["images"] == image_name].copy()

    if image_data.empty:
        raise ValueError(f"No data found for image: {image_name}")

    print(f"Found {len(image_data)} ground truth annotations for {image_name}")
    return image_data


def ensure_centers(gt_df: pd.DataFrame) -> pd.DataFrame:
    if "x" in gt_df.columns and "y" in gt_df.columns:
        logger.info("Ground truth (x, y) annotations already there")
        return gt_df
    else:
        gt_df = gt_df.copy()
        gt_df["x"] = 0.5 * (gt_df["x_min"] + gt_df["x_max"])
        gt_df["y"] = 0.5 * (gt_df["y_min"] + gt_df["y_max"])

        return gt_df


@app.command("inference")
def inference(
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", help="path to model.pth"),
    gt_path: Path = typer.Option(
        DEFAULT_GT_PATH, "--gt-path", help="path to gt truth.csv (bbboxes)"
    ),
    img_root: Path = typer.Option(DEFAULT_IMG_ROOT, "--img-root", help="path to images root dir"),
    device: str | None = typer.Option(None, "--device", help="accelerator device to use"),
    out_path: Path = typer.Option(
        Path("./data/test_results_v2/inference.json"), "--out-path", help="path to output csv"
    ),
) -> None:
    assert out_path.suffix == ".json", "out_path must be .json"

    gt_all_imgs_df = ensure_centers(pd.read_csv(gt_path))

    unique_images = gt_all_imgs_df["images"].unique()
    print(
        f"gt_all_imgs_df {gt_all_imgs_df.shape}, unique_images: {len(unique_images)}, "
        f" columns={list(gt_all_imgs_df.columns)}"
    )

    model = im.load_trained_model(model_path)

    device = (
        device or "mps"
        if torch.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # for image in unique_images[5:]:
    collected_results = []
    # for image in ["018f5ab5b7516a47ff2ac48a9fc08353b533c30f.JPG"]:
    for image in unique_images[:2]:
        image_path = img_root / image
        gt_img_df = get_single_image_gt(gt_all_imgs_df, image)
        predictions = im.predict_single_image_v2(model=model, image_path=image_path, device=device)
        collected_results.append(
            {
                "images": image,
                "ground_truth": gt_img_df.drop("images", axis=1).to_dict(
                    orient="split", index=False
                ),
                "predictions": predictions["x,y,labels,scores".split(",")].to_dict(
                    orient="split", index=False
                ),
            }
        )
        # pprint(results)

    logger.info(f"Writing collected results ({len(collected_results)}) to: {out_path!s}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f_out:
        json.dump(collected_results, f_out, indent=4)


@app.command("eval")
def evaluate(
    inference_path: Path = typer.Option(
        Path("./data/test_results_v2/inference.json"), "--out-path", help="path to output csv"
    ),
    match_strategy: str = typer.Option("point2point"),
    match_tolerance: float | None = typer.Option(
        None,
        "--match-tolerance",
        help="tolerance for matching detections to gt, for match_strategy == 'point2point' "
        "this becomes (distance) threshold (default 5 px) ",
    ),
    out_dir: Path = typer.Option(
        Path("./data/test_results_v2/"), "--out-dir", help="path to output csv"
    ),
):
    with open(inference_path, "rb") as f_in:
        inference_results = json.load(f_in)

    if match_tolerance is None:
        match_tolerance = DEFAULT_MATCH_TOL_BY_TRAT[match_strategy]
        logger.info(f"Using match_tolerance={match_tolerance} for match_strategy={match_strategy}")

    evaluator = im.PrecisionRecallEvaluator(
        match_strategy=match_strategy,
        match_tolerance=match_tolerance,
        species_map=SPECIES_MAP,
    )

    by_image_results = []
    for inference in inference_results:
        image = inference["images"]
        gt_img_df = pd.DataFrame(**inference["ground_truth"])
        predictions = pd.DataFrame(**inference["predictions"])
        results = evaluator.evaluate_preds(ground_truth=gt_img_df, predictions=predictions)
        image_results = {
            "images": image,
            **results,
        }
        by_image_results.append(image_results)

    results = {
        "match_tolerance": match_tolerance,
        "match_strategy": match_strategy,
        "by_image_results": by_image_results,
    }

    logger.info(f"{len(inference_results)=}")

    by_img_path = out_dir / "eval_by_image.json"
    logger.info(f"Writing by_image_results results ({len(by_image_results)}) to: {by_img_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(by_img_path, "w") as f_out:
        # pprint(results)
        json.dump(results, f_out, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    app()
