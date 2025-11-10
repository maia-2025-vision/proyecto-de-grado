import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    from pprint import pformat, pprint

    import animaloc
    import torch
    from animaloc.eval.stitchers import Stitcher
    from animaloc.models.utils import LossWrapper, load_model
    from loguru import logger
    from omegaconf import DictConfig, OmegaConf
    from PIL import Image
    from torchvision.transforms import ToTensor

    from api.animaloc_utils import build_model_from_cfg
    from api.torch_utils import pick_torch_device

    return (
        DictConfig,
        Image,
        OmegaConf,
        Path,
        Stitcher,
        ToTensor,
        animaloc,
        build_model_from_cfg,
        logger,
        pformat,
        pick_torch_device,
        torch,
    )


@app.cell
def _(
    DictConfig,
    OmegaConf,
    Path,
    Stitcher,
    animaloc,
    build_model_from_cfg,
    logger,
    pformat,
    pick_torch_device,
    torch,
):
    def define_stitcher(
        model: torch.nn.Module, cfg: DictConfig, img_size: int, device_name: str
    ) -> Stitcher:
        """Make a stitcher so that we can process images larger than the size the model was trained on.

        Essentially the same as Herdnet/tools/test.py:_define_stitcher,
        only change being that img_size and device_name are passed as arguments instead taking them from config
        """
        name = cfg.name

        assert name in animaloc.eval.stitchers.__dict__.keys(), (
            f"'{name}' class unfound, make sure you have included the class in the stitchers list"
        )

        kwargs = dict(cfg.kwargs)
        for k in ["model", "size", "device_name"]:
            kwargs.pop(k, None)

        stitcher = animaloc.eval.stitchers.__dict__[name](
            model=model, size=img_size, **kwargs, device_name=device_name
        )

        return stitcher

    stitcher_cfg = DictConfig(
        content={
            "name": "HerdNetStitcher",
            "kwargs": {
                "overlap": 160,
                "down_ratio": 1,
                "up": False,
                "reduction": "mean",  # TODO should this be ?
            },
        }
    )

    device = pick_torch_device()

    print("device =", device)

    model_pth_path = "data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model-by-epoch-70.pth"

    model_cfg = DictConfig(
        content={
            "model": {
                "name": "FasterRCNNResNetFPN",
                "from_torchvision": False,
                "pth_file": model_pth_path,
                "kwargs": dict(
                    architecture="resnet50",
                    num_classes=7,
                    pretrained_backbone=True,
                    trainable_backbone_layers=4,
                    class_weights=[0.1, 1.2, 1.9, 1.16, 6.4, 12.1, 1.0],
                    min_size=512,
                    max_size=512,
                ),
            },
            "dataset": {"num_classes": 7},
        }
    )

    def make_faster_rcnn_model_from_cfg_file(
        model_pth_path: Path,
        cfg_file: Path,
    ) -> torch.nn.Module:
        cfg = OmegaConf.load(cfg_file)
        cfg.model.pth_file = model_pth_path
        logger.info(f"Excerpts from cfg:\n{pformat(cfg.model)}\n{cfg.datasets.num_classes=}")
        model = build_model_from_cfg(cfg)
        model.eval()
        device_name = pick_torch_device()
        logger.info(f"putting model on device: {device_name}")
        model.to(device_name)

        return model

    model = make_faster_rcnn_model_from_cfg_file(
        model_pth_path=model_pth_path, cfg_file="configs/train/faster_rcnn_herdnet_trainer.yaml"
    )

    return define_stitcher, device, model, stitcher_cfg


@app.cell
def _(define_stitcher, device, model, stitcher_cfg):
    stitcher = define_stitcher(model, cfg=stitcher_cfg, img_size=2000, device_name=device)
    print(stitcher)
    return


@app.cell
def _(model):
    model
    return


@app.cell
def _():
    from inspect import getsource

    from api.model_utils import make_faster_rcnn_model

    print(getsource(make_faster_rcnn_model))
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv("data/patches-512-ol-160-m0.3/train/gt.csv")
    df.groupby("images").agg({"labels": len})
    return


@app.cell
def _(Image, ToTensor):
    import numpy as np
    from albumentations.augmentations import Normalize

    # image_path = "data/train_subframes/L_07_05_16_DSC00127_S3.JPG"
    image_path = (
        "data/patches-512-ol-160-m0.3/train/006b4661847b82acfb2b6a3e3677f4ae63f1dd5c_101.JPG"
    )

    # _load_image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    # img1 = img.crop([1488, 300, 2000, 812])

    to_tensor = ToTensor()
    norm = Normalize()
    img_tr = norm(image=img_np)
    img_tr
    img1_tensor = to_tensor(img_tr["image"])
    img, img1_tensor.shape
    return Normalize, img1_tensor


@app.cell
def _(img1_tensor):
    img1_tensor.shape
    return


@app.cell
def _(device, img1_tensor, model, torch):
    with torch.no_grad():
        preds = model([img1_tensor.to(device)])

    preds[0][0], preds[0][0]["boxes"].shape
    return (preds,)


@app.cell
def _(preds):
    preds[0][0]["scores"]
    return


@app.cell
def _(Normalize, ToTensor):
    to_tensor1 = ToTensor()
    norm1 = Normalize()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
