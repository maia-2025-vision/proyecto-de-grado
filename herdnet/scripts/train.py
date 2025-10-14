import os
import yaml
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

# HerdNet modules
from herdnet.data.dataset import HerdNetDataset
from herdnet.data.transforms import get_transforms
from herdnet.engine.trainer import Trainer
from herdnet.model.herdnet import build_herdnet
from herdnet.utils.utils import set_seed, prepare_device

# ------------------------------
# Config loader
# ------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def custom_collate_fn(batch):
    # batch es una lista de diccionarios
    collated = {key: [d[key] for d in batch] for key in batch[0]}
    return collated

# ------------------------------
# Main training entry point
# ------------------------------
def main(config):

    # 1. Set seed & device
    set_seed(config['seed'])
    device = prepare_device(config['device'])

    # 2. Transforms
    train_transforms = get_transforms(mode='train', image_size=config['dataset']['train_image_size'])
    val_transforms = get_transforms(mode='val')  # no resize

    # 3. Datasets
    train_dataset = HerdNetDataset(
        annotation_path=config['dataset']['train_annotation'],
        image_dir=config['dataset']['train_image_dir'],
        image_transforms=train_transforms,
        image_size=config['dataset']['train_image_size']
    )

    val_dataset = HerdNetDataset(
        annotation_path=config['dataset']['val_annotation'],
        image_dir=config['dataset']['val_image_dir'],
        image_transforms=val_transforms,
        image_size=None  # tamaño libre
    )

    train_subset = Subset(train_dataset, range(20))
    # 4. DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['workers'],
        collate_fn=custom_collate_fn
    )

    val_subset = Subset(val_dataset, range(5))
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=config['dataset']['workers'],
        collate_fn=custom_collate_fn 
    )

    # 5. Model
    model = build_herdnet(config)
    model.to(device)

    # 6. Entrenador
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # 7. Entrenar
    trainer.train()


# ------------------------------
# CLI
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HerdNet model")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
