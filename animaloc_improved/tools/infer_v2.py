__copyright__ = """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be)
    for any questions.

    CHANGES by MAIA Team:
    1. added required -dest option to set output directory
    2. added -xplot option (defaulting to False) to control whether to export plots or not
      (plot = image with detection points drawn over it)
    3. added -xthumb option (defaulting to False) to control whether to export thumbnails or not.
    4. separated export procedure out into separate function called at the end if xplot is True.

    Last modification: Nov 3, 2025
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

# ruff: noqa: C408  # Unnecessary `dict` call (rewrite as a literal)

import argparse
import os
import warnings
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy
import pandas as pd
import PIL
import torch
from animaloc.data.transforms import DownSample, Rotate90
from animaloc.datasets import CSVDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher
from animaloc.eval.metrics import PointsMetrics
from animaloc.models import HerdNet, LossWrapper
from animaloc.vizual import draw_points, draw_text
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
PIL.Image.MAX_IMAGE_PIXELS = None


parser = argparse.ArgumentParser(
    prog="inference",
    description="Collects the detections of a pretrained HerdNet model on a set of images ",
)

parser.add_argument("root", type=str, help="path to the JPG images folder (str)")
parser.add_argument(
    "-pattern",
    type=str,
    default="**",
    help="pattern for paths relative to root to include in inference (str), "
    "e.g if root='data/patches' and ppat='patches_wo_annots/**' "
    "then only images under 'data/patches/patches_wo_annots' will be processed, "
    "but theirs paths in the output detections file will be relative to data/patches. ",
)
parser.add_argument("pth", type=str, help="path to PTH file containing your model parameters (str)")
parser.add_argument(
    "-size", type=int, default=512, help="patch size use for stitching. Defaults to 512."
)
parser.add_argument(
    "-dest", type=str, help="destination folder for saving detections", required=True
)
parser.add_argument("-over", type=int, default=160, help="overlap for stitching. Defaults to 160.")
parser.add_argument(
    "-device",
    type=str,
    default="cuda",
    help="device on which model and images will be allocated (str). \
        Possible values are 'cpu' or 'cuda'. Defaults to 'cuda'.",
)
parser.add_argument("-ts", type=int, default=256, help="thumbnail size. Defaults to 256.")
parser.add_argument("-pf", type=int, default=10, help="print frequency. Defaults to 10.")
parser.add_argument(
    "-rot", type=int, default=0, help="number of times to rotate by 90 degrees. Defaults to 0."
)
parser.add_argument(
    "-xplots",
    type=bool,
    default=False,
    help="whether to export plots (images with detections marked on them)",
)
parser.add_argument(
    "-xthumbs",
    type=bool,
    default=False,
    help="whether to export thumbnails (one small image centered around detection)",
)
parser.add_argument(
    "-override_labels",
    type=int,
    default=-1,
    help="whether to override predicted labels before writing the output."
    "Default -1 means no override, any value >= 0 will cause override."
    "This option is meant to be used for hard negative mining.",
)

args = parser.parse_args()


def main() -> None:
    # Create destination folder
    # curr_date = current_date()
    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    # dest = os.path.join(args.root, f"{curr_date}_HerdNet_results")

    # Load model from PTH file
    device_name = args.device
    logger.info(f"Loading model checkpoint from {args.pth}")
    checkpoint = torch.load(args.pth, map_location=torch.device(device_name))
    classes = checkpoint["classes"]
    logger.info(f"{classes=}")
    num_classes = len(classes) + 1
    img_mean = checkpoint["mean"]
    img_std = checkpoint["std"]

    # Prepare dataset and dataloader
    img_root_path = Path(args.root)
    img_names: list[str] = [
        str(rel_path.relative_to(img_root_path))
        for rel_path in img_root_path.rglob(args.pattern)
        if rel_path.suffix in (".JPG", ".jpg", ".JPEG", ".jpeg")
    ]
    n = len(img_names)
    logger.info(f"{n} images found under {img_root_path!s}")

    df = pd.DataFrame(data={"images": img_names, "x": [0] * n, "y": [0] * n, "labels": [1] * n})

    end_transforms = []
    if args.rot != 0:
        end_transforms.append(Rotate90(k=args.rot))
    end_transforms.append(DownSample(down_ratio=2, anno_type="point"))

    albu_transforms = [A.Normalize(mean=img_mean, std=img_std)]

    dataset = CSVDataset(
        csv_file=df,
        root_dir=args.root,
        albu_transforms=albu_transforms,
        end_transforms=end_transforms,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=torch.utils.data.SequentialSampler(dataset)
    )

    # Build the trained model
    logger.info(f"Building HerdNet model with {num_classes} classes")
    model = HerdNet(num_classes=num_classes, pretrained=False)
    model = LossWrapper(model, [])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Build the evaluator
    stitcher = HerdNetStitcher(
        model=model,
        size=(args.size, args.size),
        overlap=args.over,
        down_ratio=2,
        up=True,
        reduction="mean",
        device_name=device_name,
    )

    metrics = PointsMetrics(5, num_classes=num_classes)
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        lmds_kwargs=dict(kernel_size=(3, 3), adapt_ts=0.2, neg_ts=0.1),  # noqa: C408
        device_name=device_name,
        print_freq=args.pf,
        stitcher=stitcher,
        work_dir=str(dest_path),
        header="[INFERENCE]",
    )

    # Start inference
    logger.info("Starting inference ...")
    evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)

    # Save the detections
    logger.info(f"Saving the detections to: {dest_path!s}/detections.csv")
    detections = evaluator.detections
    detections.dropna(inplace=True)
    detections["species"] = detections["labels"].map(classes)
    if args.override_labels > -1:
        logger.info(f"Overriding labels to {args.override_labels}")
        detections["labels"] = args.override_labels
        detections["species"] = "(detected label overriden)"

    detections.to_csv(dest_path / "detections.csv", index=False)

    # Draw detections on images and create thumbnails

    dest_thumb: Path | None = None
    if args.xthumbs:  # do export thumbnails
        dest_thumb = dest_path / "thumbnails"
        dest_thumb.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Not exporting thumbnails.")

    if args.xplots:
        dest_plots = dest_path / "plots"
        dest_plots.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting plots ...")
        export_plots(detections, dest_plots, dest_thumb)
    else:
        logger.info("Not exporting plots.")


def export_plots(detections: pd.DataFrame, dest_plots: Path, dest_thumb: Path | None) -> None:
    img_names = numpy.unique(detections["images"].values).tolist()  # type: ignore [arg-type]
    for img_name in img_names:
        img = Image.open(os.path.join(args.root, img_name))
        if args.rot != 0:
            rot = args.rot * 90
            img = img.rotate(rot, expand=True)  # type: ignore[assignment]
        img_cpy = img.copy()
        pts = list(detections[detections["images"] == img_name][["y", "x"]].to_records(index=False))
        pts = list(pts)  # CHANGE by aalea prev [(y, x) for y, x in pts]
        output = draw_points(img, pts, color="red", size=10)

        output.save(dest_plots / img_name, quality=95)

        # Create and export thumbnails
        if dest_thumb is None:
            # Do not export thumbnails
            continue

        sp_score = list(
            detections[detections["images"] == img_name][["species", "scores"]].to_records(
                index=False
            )
        )
        for i, ((y, x), (sp, score)) in enumerate(zip(pts, sp_score, strict=False)):
            off = args.ts // 2
            coords = (x - off, y - off, x + off, y + off)
            thumbnail = img_cpy.crop(coords)
            score = round(score * 100, 0)
            thumbnail = draw_text(
                thumbnail, f"{sp} | {score}%", position=(10, 5), font_size=int(0.08 * args.ts)
            )
            thumbnail.save(dest_thumb / img_name[:-4] + f"_{i}.JPG")


if __name__ == "__main__":
    main()
