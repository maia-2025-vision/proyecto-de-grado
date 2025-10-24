from pathlib import Path
from pprint import pformat

import animaloc
import numpy as np
import torch
import torchvision
from albumentations.augmentations import Normalize
from animaloc.eval.stitchers import Stitcher
from animaloc.models.utils import LossWrapper, load_model
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from api.detector import Detector, RawDetections
from api.internal_types import BBoxFormat, ModelMetadata
from api.torch_utils import pick_torch_device


def build_model_from_cfg(cfg: DictConfig) -> torch.nn.Module:
    """Build a model from a config.

    Same code as Herdnet/tools/test.py:_build_model

    Only keys used from cfg are:
      - cfg.model.name,  e.g
      - cfg.model.from_torchvision, e.g False
      - cfg.model.kwargs
      - cfg.dataset.num_classes

    """
    name = cfg.model.name
    from_torchvision = cfg.model.from_torchvision

    if from_torchvision:
        assert name in torchvision.models.__dict__.keys(), (
            f"'{name}' unfound in torchvision's models"
        )

        model = torchvision.models.__dict__[name]

    else:
        assert name in animaloc.models.__dict__.keys(), (
            f"'{name}' class unfound,  make sure you have included the class in the models list"
        )

        model = animaloc.models.__dict__[name]

    kwargs = dict(cfg.model.kwargs)
    for k in ["num_classes"]:
        kwargs.pop(k, None)

    model = model(**kwargs, num_classes=cfg.datasets.num_classes)
    model = LossWrapper(model, [])
    model = load_model(model, cfg.model.pth_file)
    assert isinstance(model, torch.nn.Module), f"{type(model).__name__}="

    logger.info(f"loaded model from: {cfg.model.pth_file}")
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
        norm: Normalize,
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

        self.stitcher = Stitcher(
            model=model,
            size=(patch_size, patch_size),
            overlap=100,
            batch_size=batch_size,
            device_name=device_name,
        )

    def get_idx_2_species_dict(self) -> dict[int, str]:
        """Get the mapping from idx to species name."""
        return self.idx2species

    def detect_one_image(self, image: Image.Image) -> RawDetections:
        """Get all detections on one image."""
        img = image.convert("RGB")
        img_np = np.array(img)
        img_tr = self.norm(image=img_np)
        img_tensor = self.to_tensor(img_tr["image"])

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


def faster_rcnn_detector_from_cfg_file(
    model_pth_path: Path,
    cfg_file: Path,
) -> tuple[Detector, ModelMetadata]:
    cfg = OmegaConf.load(cfg_file)
    cfg.model.pth_file = model_pth_path

    logger.info(f"Excerpts from cfg:\n{pformat(dict(cfg.model))}\n{cfg.datasets.num_classes=}")

    # Build model, set to eval mode and load onto device
    assert isinstance(cfg, DictConfig), f"{type(cfg).__name__=}, should be a DictConfig..."
    model = build_model_from_cfg(cfg)
    model.eval()
    device_name = pick_torch_device()
    logger.info(f"putting model on device: {device_name}")
    model.to(device_name)

    assert cfg.model.kwargs.min_size == cfg.model.kwargs.max_size, (
        f"Expected {cfg.model.kwargs.min_size} == {cfg.model.kwargs.max_size}"
    )
    patch_size = cfg.model.kwargs.max_size
    detector = FasterRCNNDetector(
        model=model,
        norm=Normalize(),  # BUILDING transform with default mean/std params for now...
        patch_size=patch_size,
        batch_size=cfg.inference_settings.batch_size,
        bbox_format="xyxy",
        idx2species=dict(cfg.datasets.class_def),
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
