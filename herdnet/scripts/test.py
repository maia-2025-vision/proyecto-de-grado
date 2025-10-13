import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from herdnet.config.config_parser import load_config
from herdnet.model.herdnet import build_herdnet
from herdnet.data.dataset import HerdNetDataset
from herdnet.evaluation.evaluator import extract_points_from_density_map, match_detections_to_gt

def test_model(config_path, checkpoint_path):
    config = load_config(config_path)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # ------------------------
    # 1. Dataset (Test set)
    # ------------------------
    test_dataset = HerdNetDataset(
        annotation_file=config['dataset']['test_annotation'],
        image_dir=config['dataset']['test_image_dir'],
        image_size=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['dataset']['workers']
    )

    # ------------------------
    # 2. Modelo
    # ------------------------
    model = build_herdnet(config)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[INFO] Probando modelo desde checkpoint: {checkpoint_path}")
    print(f"[INFO] Total de imágenes en test set: {len(test_dataset)}")

    # ------------------------
    # 3. Evaluación
    # ------------------------
    total_TP, total_FP, total_FN = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            image = batch['image'][0].to(device)  # [3, H, W]
            centers = batch['centers'][0]         # Tensor [N, 2]
            classes = batch['classes'][0]         # Tensor [N]

            output = model(image.unsqueeze(0))    # [1, C, H, W]
            pred_points = extract_points_from_density_map(
                output[0], threshold=0.2
            )

            gt_points = [(int(x.item()), int(y.item()), int(c.item()))
                         for (x, y), c in zip(centers, classes)]

            TP, FP, FN = match_detections_to_gt(
                pred_points, gt_points,
                max_dist=config['evaluation']['max_detection_distance']
            )

            total_TP += TP
            total_FP += FP
            total_FN += FN

    precision = total_TP / (total_TP + total_FP + 1e-8)
    recall = total_TP / (total_TP + total_FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print("\n====== Resultados Finales del Test ======")
    print(f"TP: {total_TP} | FP: {total_FP} | FN: {total_FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("=========================================")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test de modelo HerdNet")
    parser.add_argument('--config', type=str, required=True,
                        help="Ruta al archivo de configuración YAML")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Ruta al checkpoint del modelo .pth")
    args = parser.parse_args()

    test_model(args.config, args.checkpoint)
