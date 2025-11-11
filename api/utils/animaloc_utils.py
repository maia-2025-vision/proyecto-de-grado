from collections.abc import Mapping
from pathlib import Path
from pprint import pformat
from typing import Any

import albumentations.augmentations as aa
import animaloc
import numpy as np
import torch
import torchvision
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.models.utils import LossWrapper
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from animaloc_improved.eval.stitchers import FasterRCNNStitcherV2
from api.detector import Detector, RawDetections
from api.schemas.internal_types import BBoxFormat, ModelMetadata


def pick_inference_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_checkpoint(model: torch.nn.Module, pth_path: str) -> torch.nn.Module:
    """Load model parameters from a PTH file

    Args:
        model (torch.nn.Module): the model
        pth_path (str): path to the PTH file containing model parameters

    Returns:
        torch.nn.Module
            the model with loaded parameters
    """
    map_location = torch.device("cpu")
    if torch.cuda.is_available():
        map_location = torch.device("cuda")

    return torch.load(pth_path, map_location=map_location)


def build_model_from_cfg(cfg: DictConfig, model_state_dict: Mapping[str, Any]) -> torch.nn.Module:
    """Build a model from a config and a state dict

    Very similar Herdnet/tools/test.py:_build_model

    Only keys used from cfg are:
      - cfg.model.name
      - cfg.model.from_torchvision, e.g False
      - cfg.model.kwargs
      - cfg.dataset.num_classes
    """
    name = cfg.model.name
    from_torchvision = cfg.model.from_torchvision

    if from_torchvision:
        assert name in torchvision.models.__dict__.keys(), (
            f"'{name}' NOT found in torchvision's models"
        )

        model = torchvision.models.__dict__[name]

    else:
        assert name in animaloc.models.__dict__.keys(), (
            f"'{name}' class NOT found,  make sure you have included the class in the models list"
        )

        model = animaloc.models.__dict__[name]

    kwargs = dict(cfg.model.kwargs)
    for k in ["num_classes"]:
        kwargs.pop(k, None)

    logger.info(f"Building model with kwargs: {kwargs}, num_classes: {cfg.dataset.num_classes}")
    model = model(**kwargs, num_classes=cfg.dataset.num_classes)
    model = LossWrapper(model, [])
    # model = load_model(model, cfg.model.pth_file)
    model.load_state_dict(model_state_dict)
    assert isinstance(model, torch.nn.Module), f"{type(model).__name__}="

    logger.info(f"loaded model of type ({type(model.model).__name__}) from: {cfg.model.pth_file}")
    return model


class FasterRCNNDetector(torch.nn.Module, Detector):
    """Detector that uses an underlying Faster R-CNN model."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        *,
        model: torch.nn.Module,
        idx2species: dict[int, str],
        patch_size: int,
        batch_size: int,
        norm: aa.Normalize,
        device_name: str,
        bbox_format: BBoxFormat,
    ) -> None:
        super().__init__()
        self.model = model
        self.idx2species = idx2species
        self.norm = norm
        self.device_name = device_name
        self.bbox_format_ = bbox_format
        self.to_tensor = ToTensor()

        self.stitcher = FasterRCNNStitcherV2(
            model=model,
            size=(patch_size, patch_size),
            overlap=100,
            batch_size=batch_size,
            device_name=device_name,
            # TODO: should we override these at prediction time?
            score_threshold=0.0,
            nms_threshold=0.5,
        )

    def get_idx_2_species_dict(self) -> dict[int, str]:
        """Get the mapping from idx to species name."""
        return self.idx2species

    def detect_one_image(self, image: Image.Image) -> RawDetections:
        """Get all detections on one image of any size, by using the appropriate stitcher"""
        img_np = np.array(image.convert("RGB"))
        img_normalized = self.norm(image=img_np)
        img_pt = self.to_tensor(img_normalized["image"])
        logger.info(f"img_pt: {img_pt.shape}")
        # No need to move to device_name, stitcher does it internally
        preds = self.stitcher(img_pt)

        return preds  # type: ignore[no-any-return]

    def detect_one_img_patch_size(self, image: Image.Image) -> RawDetections:
        """Get all detections on one image, assuming its size is exactly equal to patch size.

        That is avoid using stitcher
        """
        img = image.convert("RGB")
        img_np = np.array(img)
        img_normalized = self.norm(image=img_np)
        img_tensor = self.to_tensor(img_normalized["image"])

        img_tensor = img_tensor.unsqueeze(0).to(self.device_name)
        assert img_tensor.dim() == 4, f"{img_tensor.shape} expected to have len=4"

        with torch.no_grad():
            pred, _ = self.model(img_tensor)

        pred0 = pred[0]
        assert isinstance(pred0, dict), f"{pred0} should be a dict\npred={pformat(pred0)}"

        return pred0  # type: ignore[return-value]

    def bbox_format(self) -> BBoxFormat:
        """What is the bbox_format of my detections?"""
        return self.bbox_format_


class HerdnetDetector(torch.nn.Module, Detector):
    """Detector that uses an underlying Herdnet model."""

    def __init__(  # type: ignore[no-any-unimported]
        self,
        *,
        model: torch.nn.Module,
        idx2species: dict[int, str],
        patch_size: int,  # e.g. 512
        batch_size: int,
        normalizer: aa.Normalize,
        device_name: str,
        lmds_kwargs: Mapping[str, Any],
        # bbox_format: BBoxFormat,
    ) -> None:
        super().__init__()
        self.model = model
        self.idx2species = idx2species
        self.normalizer = normalizer
        self.device_name = device_name
        # self.bbox_format_ = bbox_format
        self.to_tensor = ToTensor()

        self.stitcher = HerdNetStitcher(
            model=model,
            device_name=device_name,
            size=(patch_size, patch_size),
            overlap=100,
            batch_size=batch_size,
            # TODO: should we override these at prediction time?
            # Taken from animaloc/tools/infer.py
            down_ratio=2,
            up=True,
            reduction="mean",
        )

        up = self.stitcher is None
        self.lmds = HerdNetLMDS(up=up, **lmds_kwargs)

    def get_idx_2_species_dict(self) -> dict[int, str]:
        """Get the mapping from idx to species name."""
        return self.idx2species

    def detect_one_image(self, image: Image.Image) -> RawDetections:
        """Get all detections on one image of any size, by using the appropriate stitcher"""
        img_np = np.array(image.convert("RGB"))
        img_normalized = self.normalizer(image=img_np)
        img_pt = self.to_tensor(img_normalized["image"])
        logger.info(f"img_pt: {img_pt.shape}")
        # No need to move to device_name, stitcher does it internally
        density_maps = self.stitcher(img_pt)
        logger.info(f"preds: {density_maps.shape}")

        heatmap = density_maps[:, :1, :, :]
        clsmap = density_maps[:, 1:, :, :]
        logger.info(f"heatmap: {heatmap.shape}, clsmap: {clsmap.shape}")
        counts, points, labels, scores, dscores = self.lmds((heatmap, clsmap))

        preds = dict(
            points=torch.tensor(points[0]),
            labels=torch.tensor(labels[0]),
            scores=torch.tensor(scores[0]),
            dscores=torch.tensor(dscores[0]),
        )
        # logger.info(f"preds: {preds}")

        return preds  # type: ignore[no-any-return]

    def detect_one_img_patch_size(self, image: Image.Image) -> RawDetections:
        """Get all detections on one image, assuming its size is exactly equal to patch size.

        That is avoid using stitcher
        """
        img = image.convert("RGB")
        img_np = np.array(img)
        img_normalized = self.norm(image=img_np)
        img_tensor = self.to_tensor(img_normalized["image"])

        img_tensor = img_tensor.unsqueeze(0).to(self.device_name)
        assert img_tensor.dim() == 4, f"{img_tensor.shape} expected to have len=4"

        with torch.no_grad():
            pred, _ = self.model(img_tensor)

        pred0 = pred[0]
        assert isinstance(pred0, dict), f"{pred0} should be a dict\npred={pformat(pred0)}"

        return pred0  # type: ignore[return-value]

    def bbox_format(self) -> BBoxFormat | None:
        """What is the bbox_format of my detections?"""
        return None


def faster_rcnn_detector_from_cfg_file(
    model_pth_path: Path,
    cfg_file: Path,
) -> tuple[Detector, ModelMetadata]:
    cfg = OmegaConf.load(cfg_file)
    cfg.model.pth_file = model_pth_path

    logger.info(f"Excerpts from cfg:\n{pformat(dict(cfg.model))}\n{cfg.dataset.num_classes=}")

    # Build model, set to eval mode and load onto device
    assert isinstance(cfg, DictConfig), f"{type(cfg).__name__=}, should be a DictConfig..."
    device_name = pick_inference_device()
    logger.info(f"will load checkpoint onto device={device_name}")
    checkpoint = torch.load(model_pth_path, map_location=device_name)
    model = build_model_from_cfg(cfg, model_state_dict=checkpoint["model_state_dict"])
    model.eval()

    assert cfg.model.kwargs.min_size == cfg.model.kwargs.max_size, (
        f"Expected {cfg.model.kwargs.min_size} == {cfg.model.kwargs.max_size}"
    )
    patch_size = cfg.model.kwargs.max_size
    detector = FasterRCNNDetector(
        model=model,
        norm=aa.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std),
        patch_size=patch_size,
        # Se fija batch_size en 1, el pipeline actual no soporta lotes > 1.
        batch_size=1,  # cfg.inference_settings.batch_size,
        bbox_format="xyxy",
        idx2species=dict(cfg.dataset.class_def),
        device_name=device_name,
    )

    metadata = {
        "name": cfg.model.name,
        "class_name": type(model).__name__,
        "backbone_arch": cfg.model.kwargs.architecture,
        "patch_size": patch_size,
        "trainable_backbone_layers": cfg.model.kwargs.trainable_backbone_layers,
        "num_classes": cfg.model.kwargs.num_classes,
        "weights_path": model_pth_path,
        "cfg_path": cfg_file,
    }

    return detector, metadata


def herdnet_detector_from_cfg_file(
    model_pth_path: Path,
    cfg_file: Path,
) -> tuple[Detector, ModelMetadata]:
    cfg = OmegaConf.load(cfg_file)
    cfg.model.pth_file = model_pth_path

    logger.info(f"Excerpts from cfg:\n{pformat(dict(cfg.model))}\n{cfg.dataset.num_classes=}")

    # Build model, set to eval mode and load onto device
    assert isinstance(cfg, DictConfig), f"{type(cfg).__name__=}, should be a DictConfig..."
    checkpoint_path = cfg.model.pth_file
    device_name = pick_inference_device()
    logger.info(f"Will load model onto device: {device_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device_name)
    model = build_model_from_cfg(cfg, model_state_dict=checkpoint["model_state_dict"])
    model.eval()

    patch_size = cfg.dataset.img_size[0]
    detector = HerdnetDetector(
        model=model,
        normalizer=aa.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        ),
        patch_size=patch_size,
        # Se fija batch_size en 1, el pipeline actual no soporta lotes > 1.
        batch_size=1,
        idx2species=dict(cfg.dataset.class_def),
        device_name=device_name,
        # Values for lmds_kwargs taken from animaloc/tools/infer.py
        lmds_kwargs=dict(kernel_size=(3, 3), adapt_ts=0.2, neg_ts=0.1),
    )
    # logger.info(f"built detector={detector}")

    metadata = {
        "name": cfg.model.name,
        "class_name": type(model).__name__,
        "patch_size": patch_size,
        "num_classes": cfg.dataset.num_classes,
        "weights_path": model_pth_path,
        "cfg_path": cfg_file,
    }
    logger.info(f"metadata={metadata}")

    return detector, metadata
