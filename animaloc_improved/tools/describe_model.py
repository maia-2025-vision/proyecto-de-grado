from pathlib import Path
from pprint import pformat

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from oficial_herdnet.tools.test import build_model


def describe_nn_module(model: nn.Module, group_depth: int = 2) -> pd.DataFrame:
    param_data = []

    for name, param in model.named_parameters():
        # if param.requires_grad:
        name_parts = name.split(".")
        group = ".".join(name_parts[:group_depth])
        param_data.append(
            {
                "group": group,
                "name": name,
                "trainable": param.requires_grad,
                "num_params_MM": param.numel() / 1e6,
                "shape": tuple(param.shape),
            }
        )

    return pd.DataFrame(param_data)


@hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    cfg = cfg.test
    logger.info(f"Full test config:\n{pformat(dict(cfg))}")

    # Prepare dataset and dataloader
    # Build the trained model
    logger.info(f"Building the trained model (on device={cfg.device_name}) ...")
    model = build_model(cfg).to(cfg.device_name)

    group_depth = cfg.get("group_depth", 2)
    desc = describe_nn_module(model, group_depth=group_depth)
    if group_depth > 0:
        desc = (
            desc.groupby(["group", "trainable"])
            .agg(
                {
                    "num_params_MM": "sum",
                }
            )
            .fillna(0)
        )

    results_file = Path(cfg.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)

    out_fmt = results_file.suffix
    if out_fmt == ".csv":
        desc.to_csv(results_file)
    elif out_fmt == ".md":
        desc.to_markdown(results_file)
    else:
        raise ValueError(f"Unknown output format: {out_fmt}")

    print(desc.to_markdown())

    logger.info(f"Results saved to {results_file!s}")


if __name__ == "__main__":
    main()
