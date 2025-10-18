import os
import warnings
import argparse
import yaml
from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT
from animaloc.data.batch_utils import collate_fn
from animaloc.models import HerdNet, LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from torch import Tensor
import albumentations as A

# Silenciar warnings específicos
def configure_warnings(verbose=False):
    """Configura warnings para el entrenamiento"""
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="animaloc.eval.metrics")
        warnings.filterwarnings("ignore", message=".*Found torch version.*contains a local version label.*")
        warnings.filterwarnings("ignore", message=".*Failed to resolve installed pip version.*")
        warnings.filterwarnings("ignore", message=".*Model logged without a signature.*")

configure_warnings(verbose=False)  # Cambiar a True si quieres ver todos los warnings

# Load configurations from train.yml
config_path = os.path.join(os.path.dirname(__file__), 'train.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations
hyperparams = config['hyperparameters']
trainer_config = config['trainer']  # Renombrar para evitar conflicto
model_config = config['model']

# Guardar configuraciones en un archivo de reporte
def save_config_report(config_dict, work_dir):
    report_path = os.path.join(work_dir, 'training_config_report.txt')
    with open(report_path, 'w') as f:
        f.write("==========================================\n")
        f.write("      HERDNET TRAINING CONFIGURATION      \n")
        f.write("==========================================\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETERS:\n")
        f.write("-----------------\n")
        for key, value in config_dict['hyperparameters'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Model configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write("-------------------\n")
        for key, value in config_dict['model'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Trainer configuration
        f.write("TRAINER CONFIGURATION:\n")
        f.write("---------------------\n")
        f.write(f"csv_logger: {config_dict['trainer']['csv_logger']}\n")
        f.write("\nPaths:\n")
        for key, value in config_dict['trainer']['paths'].items():
            f.write(f"  {key}: {value}\n")
        
        # Dataset info
        f.write("\nDATASET INFO:\n")
        f.write("------------\n")
        import pandas as pd
        try:
            train_df = pd.read_csv(os.path.abspath(config_dict['trainer']['paths']['train_csv']))
            f.write(f"Training samples: {len(train_df)}\n")
            f.write(f"Unique images: {train_df['images'].nunique()}\n")
            f.write(f"Annotations per image: {len(train_df)/train_df['images'].nunique():.2f}\n")
        except Exception as e:
            f.write(f"Error reading training dataset: {str(e)}\n")
        
        # Transforms
        f.write("\nTRANSFORMS:\n")
        f.write("-----------\n")
        f.write("Training transforms:\n")
        f.write("  - VerticalFlip(p=0.5)\n")
        f.write("  - Normalize(p=1.0)\n")
        f.write("  - FIDT(num_classes={}, down_ratio={})\n".format(
            config_dict['model']['num_classes'], 
            config_dict['model']['down_ratio']
        ))
        f.write("  - PointsToMask(radius=2, num_classes={}, squeeze=True, down_ratio=32)\n\n".format(
            config_dict['model']['num_classes']
        ))
        
        # Hardware info
        f.write("\nHARDWARE INFO:\n")
        f.write("-------------\n")
        import torch
        f.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
        
        # Date and time
        from datetime import datetime
        f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==========================================\n")
    
    print(f"Configuration report saved to: {report_path}")

def main():
    # Trainer configuration (paths and settings)
    train_csv = os.path.abspath(trainer_config['paths']['train_csv'])
    val_csv = os.path.abspath(trainer_config['paths']['val_csv'])
    train_root = os.path.abspath(trainer_config['paths']['train_root'])
    val_root = os.path.abspath(trainer_config['paths']['val_root'])
    work_dir = os.path.abspath(trainer_config['paths']['work_dir'])
    csv_logger = bool(trainer_config['csv_logger'])
    
    # Crear directorio de trabajo si no existe
    os.makedirs(work_dir, exist_ok=True)
    print(f"Working directory: {work_dir}")
    
    # Hyperparameters
    num_classes = model_config['num_classes']
    down_ratio = model_config['down_ratio']
    batch_size = int(hyperparams['batch_size'])
    lr = float(hyperparams['learning_rate'])
    weight_decay = float(hyperparams['weight_decay'])
    epochs = int(hyperparams['epochs'])

    # Guardar el reporte de configuración
    save_config_report(config, work_dir)

    # Datasets
    train_dataset = CSVDataset(
        csv_file=train_csv,
        root_dir=train_root,
        albu_transforms=[
            A.VerticalFlip(p=0.5),
            A.Normalize(p=1.0)
        ],
        end_transforms=[MultiTransformsWrapper([
            FIDT(num_classes=num_classes, down_ratio=down_ratio),
            PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=32)
        ])]
    )

    val_dataset = CSVDataset(
        csv_file=val_csv,
        root_dir=val_root,
        albu_transforms=[A.Normalize(p=1.0)],
        end_transforms=[DownSample(down_ratio=down_ratio, anno_type='point')]
    )

    # Dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # Model
    herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).cuda()

    # Losses
    weight = Tensor([0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda()
    losses = [
        {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
        {'loss': CrossEntropyLoss(reduction='mean', weight=weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]
    herdnet = LossWrapper(herdnet, losses=losses)

    # Optimizer
    optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

    # Metrics, Stitcher, Evaluator
    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    
    evaluator = HerdNetEvaluator(
        model=herdnet,
        dataloader=val_dataloader,
        metrics=metrics,
        work_dir=work_dir,
        header='validation'
    )

    # Trainer
    trainer = Trainer(
        model=herdnet,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=epochs,
        evaluator=evaluator,
        work_dir=work_dir,
        csv_logger=csv_logger
    )

    # Start training
    trainer.start(warmup_iters=100, checkpoints='best', select='max', validate_on='f1_score')

if __name__ == "__main__":
    main()