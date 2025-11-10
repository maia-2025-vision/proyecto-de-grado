import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import argparse
    import os
    from pathlib import Path

    import cv2
    import numpy
    import pandas as pd
    import PIL
    import torchvision
    from albumentations import PadIfNeeded
    from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images
    from loguru import logger
    from tqdm import tqdm

    parser = argparse.ArgumentParser(prog="patcher", description="Cut images into patches")

    parser.add_argument("root", type=str, help="path to the images directory (str)")
    parser.add_argument("height", type=int, help="height of the patches, in pixels (int)")
    parser.add_argument("width", type=int, help="width of the patches, in pixels (int)")
    parser.add_argument("overlap", type=int, help="overlap between patches, in pixels (int)")
    parser.add_argument("dest", type=str, help="destination path (str)")
    parser.add_argument(
        "-csv", type=str, help="path to a csv file containing annotations (str). Defaults to None"
    )
    parser.add_argument(
        "-min",
        type=float,
        default=0.1,
        help="minimum fraction of area for an annotation to be kept (float). Defaults to 0.1",
    )
    parser.add_argument(
        "-all",
        type=bool,
        default=False,
        help="set to True to save all patches, not only those containing annotations (bool). "
        "Defaults to False",
    )
    parser.add_argument(
        "-fpwa",
        type=float,
        default=0.0,
        help="fraction of *p*atches *w*ithout *a*nnotations to output. "
        "If -all option is also given, this value will be overriden to be 1.0.",
    )

    args = parser.parse_args(
        args=[
            "data/test",
            "512",
            "512",
            "160",
            "data/tmp",
            "-csv",
            "data/groundtruth/csv/test_big_size_A_B_E_K_WH_WB-fixed-header.csv",
            "-min",
            "0.1",
            "-fpwa",
            "0.1",
        ]
    )

    print(args)
    return PatchesBuffer, Path, args, os


@app.cell
def _(Path, args, os):
    images_paths = [
        os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith(".csv")
    ]

    dest_dir_path = Path(args.dest)
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"images-paths: {len(images_paths)}")
    print(f"{dest_dir_path=!s}")
    return


@app.cell
def _(PatchesBuffer, args, os):
    patches_buffer = PatchesBuffer(
        args.csv,
        args.root,
        (args.height, args.width),
        overlap=args.overlap,
        min_visibility=args.min,
    ).buffer
    patches_buffer.drop(columns="limits").to_csv(os.path.join(args.dest, "gt.csv"), index=False)
    return (patches_buffer,)


@app.cell
def _(patches_buffer):
    patches_buffer
    return


@app.cell
def _(patches_buffer):
    patches_buffer.iloc[0]
    return


@app.cell
def _(patches_buffer):
    patches_buffer.iloc[1]
    return


@app.cell
def _():
    5440 - 4928, 3670 - 3168
    return


@app.cell
def _(patches_buffer):
    patches_buffer.shape
    return


@app.cell
def _(patches_buffer):
    patches_buffer["images"].iloc[0:20].to_list()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
