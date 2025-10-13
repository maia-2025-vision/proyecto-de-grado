import os
import time
from herdnet.engine.evaluator import extract_points_from_density_map, match_detections_to_gt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        # Optimizer y scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma']
        )

        self.criterion = mse_loss  # se usa para el mapa de densidad

        # Checkpoints
        self.checkpoint_dir = config['output']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Cargar desde checkpoint si aplica
        ckpt_path = config['training'].get('resume_from_checkpoint')
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def train(self):
        print(f"[INFO] Iniciando entrenamiento por {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            print(f"  ➤ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.scheduler.step()

            if epoch % self.config['output']['save_freq'] == 0:
                self._save_checkpoint(epoch)

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            images_tuple, centers_tuple, classes_tuple, image_ids_tuple, orig_sizes_tuple = batch

            # Convertir imágenes a un tensor batch
            images = torch.stack(images_tuple).to(self.device)  # [B, 3, H, W]

            # centers_tuple es una tupla de tensores con diferente tamaño (por imagen)
            centers = centers_tuple  # puedes usarlos tal cual, por ejemplo para generar mapas de densidad

            bs, _, h, w = images.shape

            # Generar ground truth density maps
            gt_density = self._generate_density_map(centers, h, w).to(self.device)  # [B, C, H, W]

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, gt_density)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        running_loss = 0.0

        total_TP = 0
        total_FP = 0
        total_FN = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = torch.stack(batch['image']).to(self.device)   # [B, 3, H, W]
                centers_batch = batch['centers']
                classes_batch = batch['classes']
                bs, _, h, w = images.shape

                outputs = self.model(images)  # [B, C, H, W]

                for i in range(bs):
                    pred_density = outputs[i]  # [C, H, W]
                    pred_points = extract_points_from_density_map(pred_density, threshold=0.2)

                    gt_centers = centers_batch[i]  # Tensor [N, 2]
                    gt_classes = classes_batch[i]  # Tensor [N]
                    gt_points = [(int(x.item()), int(y.item()), int(c.item()))
                                for (x, y), c in zip(gt_centers, gt_classes)]

                    # Matching detecciones | ground truth
                    TP, FP, FN = match_detections_to_gt(pred_points, gt_points,
                                                        max_dist=self.config['evaluation']['max_detection_distance'])

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                    # Loss por imagen
                    gt_density = self._generate_density_map(gt_centers, h, w).to(self.device)
                    loss = self.criterion(pred_density.unsqueeze(0), gt_density.unsqueeze(0))  # añadir dimensión batch
                    running_loss += loss.item()

        # Métricas globales
        precision = total_TP / (total_TP + total_FP + 1e-8)
        recall = total_TP / (total_TP + total_FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"[Validación Epoch {epoch}]")
        print(f"  Loss     : {running_loss / len(self.val_loader):.4f}")
        print(f"  TP: {total_TP} | FP: {total_FP} | FN: {total_FN}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")

        return running_loss / len(self.val_loader)

    def _generate_density_map(self, batch_centers, height, width):
        """
        Genera un mapa de densidad para cada imagen del batch,
        ubicando un valor gaussiano en cada centro.
        """
        maps = []
        for centers in batch_centers:
            density = torch.zeros((self.config['dataset']['num_classes'], height, width))
            for i, (x, y) in enumerate(centers):
                x = int(round(x.item()))
                y = int(round(y.item()))
                if 0 <= x < width and 0 <= y < height:
                    # se puede cambiar por un valor gaussiano
                    density[:, y, x] += 1.0
            maps.append(density)

        return torch.stack(maps)  # [B, C, H, W]

    def _save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, ckpt_path)
        print(f"[INFO] Checkpoint guardado: {ckpt_path}")

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"[WARN] Checkpoint {path} no encontrado. Entrenamiento desde cero.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] Checkpoint cargado desde {path}")
