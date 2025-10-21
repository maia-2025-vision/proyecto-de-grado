import albumentations as alb
import torch
import hydra
from omegaconf import DictConfig
from loguru import logger

from api.model_utils import pick_torch_device


def _load_albu_transforms(tr_cfg: dict) -> list:
    transforms = []
    for name, kwargs in tr_cfg.items():
        transforms.append(alb.__dict__[name](**kwargs))

    return transforms


@hydra.main(config_path='./configs', config_name="config")
def main(cfg: DictConfig) -> None:
    """Add classes and normalize-transform parameters to a model, in-place!.

    Parameters are taken from its config.

    This is based on the last few lines of Delplanque's HerdNet/tools/train.py script.
    """
    train_cfg = cfg.train
    full_model_path = cfg.add_info.best_model_pth_path

    logger.info(f"Loading input model base from {full_model_path}, modification will be done IN PLACE")

    pth_file = torch.load(full_model_path, map_location=pick_torch_device())
    norm_trans = _load_albu_transforms(train_cfg.datasets.train.albu_transforms)[-1]

    logger.info(f"Normalization object: {norm_trans}")
    pth_file['classes'] = dict(train_cfg.datasets.class_def)

    logger.info(f"classes: {pth_file['classes']}")
    pth_file['mean'] = list(norm_trans.mean)
    pth_file['std'] = list(norm_trans.std)

    torch.save(pth_file, full_model_path)


if __name__ == "__main__":
    main()
