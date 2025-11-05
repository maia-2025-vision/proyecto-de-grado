__copyright__ = """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    This script is a slightly modified version of the original code at:
    https://github.com/Alexandre-Delplanque/HerdNet/blob/main/tools/patcher.py

    CHANGES:
      1.  Added line to create directory args.dest at startup.
      2. Admite un nuevo argumento fpwa (fraction of patches without annotations)
      (número entre 0 y 1.0) que controla la probabilidad de que un patch que no tenga anotaciones
      se genere como salida.  Por ejemplo, sobre el conjunto de train con 0.1 genera ~15000
     patches sin annotaciones. Estos los ponen en un subdirectorio de la carpeta de salida
     <dest_dir>/patches_wo_annots y la lista de las rutas a todos estos archivos queda en un csv
     <dest_dir>/gt_wo_annots.csv  (wo = without)
     3. También, el argumento -csv (anotaciones de entrada) es ahora requerido y las ramas
     de código donde este era nulo del script original se eliminaron.

    Last modification: October 21, 2025
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import argparse
import os
import random
from pathlib import Path

import cv2
import numpy
import pandas as pd
import PIL
import torch
import torchvision
from albumentations import PadIfNeeded
from animaloc.data import ImageToPatches, PatchesBuffer
from loguru import logger
from torchvision.utils import save_image
from tqdm import tqdm

parser = argparse.ArgumentParser(prog="patcher", description="Cut images into patches")

parser.add_argument("root", type=str, help="path to the images directory (str)")
parser.add_argument("height", type=int, help="height of the patches, in pixels (int)")
parser.add_argument("width", type=int, help="width of the patches, in pixels (int)")
parser.add_argument("overlap", type=int, help="overlap between patches, in pixels (int)")
parser.add_argument("dest", type=str, help="destination path (str)")
parser.add_argument(
    "-csv",
    type=str,
    help="path to a csv file containing annotations (str). Defaults to None",
    required=True,
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
parser.add_argument(
    "-seed",
    type=int,
    default=42,
    help="random seed (int). Defaults to 42",
)

args = parser.parse_args()


def save_batch_images_with_fpwa(
    batch: torch.Tensor,
    basename: str,
    dest_dir_path: Path,
    patches_with_annotations: set[str],
    fpwa: float,
) -> list[str]:
    """Save tensors into image files and also save a fraction of patches without annotations.

    This is just a modified version of save_batch_images from module animaloc.data.patches

    Use torchvision save_image function,
    see https://pytorch.org/vision/stable/utils.html#torchvision.utils.save_image

    Args:
        batch (torch.Tensor): mini-batch tensor
        basename (str) : parent image name, with extension
        dest_dir_path (str): destination folder path
        patches_with_annotations (set[str]): file names of patches with annotations
        fpwa: fraction of patches without annotations that will be saved

    Returns:
        list of patches without annotations that were saved
    """
    saved_patches_without_annotations = []
    base_wo_extension, extension = basename.split(".")[0], basename.split(".")[1]
    for i, b in enumerate(range(batch.shape[0])):
        patch_fname = f"{base_wo_extension}_{i}.{extension}"

        patch_rel_path: str | None = None  # set to a Path only for patches that will be saved
        if patch_fname in patches_with_annotations:
            patch_rel_path = patch_fname
        else:
            if random.random() < fpwa:
                patch_rel_path = f"patches_wo_annots/{patch_fname}"
                saved_patches_without_annotations.append(patch_rel_path)

        if patch_rel_path is not None:
            save_path = dest_dir_path / patch_rel_path
            save_image(batch[b], fp=save_path)

    return saved_patches_without_annotations


def main() -> None:
    if args.all:
        logger.info("-all passed, overriding value of args.fpwa to 1.0")
        args.fpwa = 1.0
    else:
        logger.info(f"Fraction of patches without annotation: {args.fpwa}")

    if args.fpwa > 0.0:
        assert args.csv is not None, "Logic for args.fpwa is only implemented when -csv is given"

    random.seed(args.seed)

    # images_paths initialized to all images
    images_paths = [
        os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith(".csv")
    ]
    logger.info(f"images_paths (all base images) has: {len(images_paths)}")

    dest_dir_path = Path(args.dest)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    (dest_dir_path / "patches_wo_annots").mkdir(parents=True, exist_ok=True)

    patches_buffer = PatchesBuffer(
        args.csv,
        args.root,
        (args.height, args.width),
        overlap=args.overlap,
        min_visibility=args.min,
    ).buffer
    logger.info(f"patches_buffer has {len(patches_buffer)} rows (individual annotations)")
    patches_with_annotations = set(patches_buffer["images"])
    logger.info(
        f"patches_buffer has {patches_buffer['images'].nunique()} unique patches with annotations"
    )
    logger.info(
        f"patches_buffer references {patches_buffer['base_images'].nunique()} "
        f"unique base images with annotations"
    )

    patches_buffer.drop(columns="limits").to_csv(dest_dir_path / "gt.csv", index=False)

    images_with_annots = set(pd.read_csv(args.csv)["images"].unique())
    logger.info(f"From csv found {len(images_with_annots)} unique (base) images with annotations")

    # if not args.all:
    if args.fpwa == 0:
        # restrict to only images without annotations
        images_paths = [os.path.join(args.root, x) for x in images_with_annots]

    logger.info(f"Exporting patches over {len(images_paths)} input image paths")

    patches_wo_annots = []
    for img_path in tqdm(images_paths, desc="Exporting patches"):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        # save some patches without annotations
        if args.fpwa > 0:
            patches = ImageToPatches(
                img_tensor, (args.height, args.width), overlap=args.overlap
            ).make_patches()
            patches_wo_annots_increment = save_batch_images_with_fpwa(
                patches,
                img_name,
                dest_dir_path=dest_dir_path,
                fpwa=args.fpwa,
                patches_with_annotations=patches_with_annotations,
            )
            patches_wo_annots.extend(patches_wo_annots_increment)

        # or only annotated ones
        else:  # args.fpwa == 0
            # position = 'top_left', updated by aalea
            padder = PadIfNeeded(
                args.height,
                args.width,
                position="top_left",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            )
            img_ptch_df = patches_buffer[patches_buffer["base_images"] == img_name]
            for row in img_ptch_df[["images", "limits"]].to_numpy().tolist():
                ptch_name, limits = row[0], row[1]
                cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                padded_img = PIL.Image.fromarray(padder(image=cropped_img)["image"])
                padded_img.save(os.path.join(args.dest, ptch_name))

    patches_wo_annots_df = pd.DataFrame({"images": patches_wo_annots})
    patches_wo_annots_path = dest_dir_path / "gt_wo_annots.csv"
    patches_wo_annots_df.to_csv(patches_wo_annots_path, index=False)
    logger.info(
        f"Saving list of {len(patches_wo_annots_df)} patches without annotations to:"
        f"{patches_wo_annots_path!s}"
    )


if __name__ == "__main__":
    main()
