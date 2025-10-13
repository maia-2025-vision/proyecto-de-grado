import os
import time
from herdnet.engine.evaluator import extract_points_from_density_map, match_detections_to_gt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss

class Trainer:
    def _prepare_density_map_batch(self, centers_list, orig_sizes, h_out, w_out):
        """
        Escala los centros de cada imagen del batch a la resolución de salida y genera el mapa de densidad.
        """
        scaled_centers = []
        for i, centers in enumerate(centers_list):
            orig_w, orig_h = orig_sizes[i]
            scale_x = w_out / orig_w
            scale_y = h_out / orig_h
            centers_i = centers.clone()
            if centers_i.numel() > 0:
                centers_i[:, 0] *= scale_x
                centers_i[:, 1] *= scale_y
            scaled_centers.append(centers_i)
        return self._generate_density_map(scaled_centers, h_out, w_out)
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

        total_batches = len(self.train_loader)
        print(f"[Entrenamiento] Epoch {epoch} - Total batches: {total_batches}")
        for batch_idx, batch in enumerate(self.train_loader):
            print(f"  [Entrenamiento] Epoch {epoch} | Batch {batch_idx+1}/{total_batches}")
            if batch_idx < 2:
                print(f"    [DEBUG] images.shape: {torch.stack(batch['image']).shape}")
                print(f"    [DEBUG] centers[0].shape: {batch['centers'][0].shape if len(batch['centers']) > 0 else 'N/A'}")
                print(f"    [DEBUG] orig_sizes[0]: {batch['orig_size'][0] if len(batch['orig_size']) > 0 else 'N/A'}")
            images = torch.stack(batch['image']).to(self.device)  # [B, 3, H, W]
            centers = batch['centers']
            orig_sizes = batch['orig_size']

            bs, _, h_in, w_in = images.shape

            outputs = self.model(images)  # [B, C, H_out, W_out]
            _, _, h_out, w_out = outputs.shape

            # Generar ground truth density maps al tamaño de salida
            gt_density = self._prepare_density_map_batch(centers, orig_sizes, h_out, w_out).to(self.device)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, gt_density)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  [Batch {batch_idx}] Loss: {loss.item():.4f}")

        return running_loss / len(self.train_loader)

    def _validate(self, epoch):
        self.model.eval()
        running_loss = 0.0

        total_TP = 0
        total_FP = 0
        total_FN = 0

        total_batches = len(self.val_loader)
        print(f"[Validación] Epoch {epoch} - Total batches: {total_batches}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                print(f"  [Validación] Epoch {epoch} | Batch {batch_idx+1}/{total_batches}")
                if batch_idx < 2:
                    print(f"    [DEBUG] images.shape: {torch.stack(batch['image']).shape}")
                    print(f"    [DEBUG] centers[0].shape: {batch['centers'][0].shape if len(batch['centers']) > 0 else 'N/A'}")
                    print(f"    [DEBUG] orig_sizes[0]: {batch['orig_size'][0] if len(batch['orig_size']) > 0 else 'N/A'}")

                images = torch.stack(batch['image']).to(self.device)   # [B, 3, H, W]
                centers_batch = batch['centers']
                orig_sizes_batch = batch['orig_size']
                bs, _, h_in, w_in = images.shape

                outputs = self.model(images)  # [B, C, H_out, W_out]
                _, _, h_out, w_out = outputs.shape

                for i in range(bs):
                    pred_density = outputs[i]  # [C, H_out, W_out]
                    pred_points = extract_points_from_density_map(pred_density, threshold=0.2)

                    gt_centers = centers_batch[i]  # Tensor [N, 2]
                    gt_classes = batch['classes'][i] if 'classes' in batch else None
                    gt_points = [(int(x.item()), int(y.item()), int(c.item()) if gt_classes is not None else 0)
                                 for (x, y), c in zip(gt_centers, gt_classes)] if gt_classes is not None else [(int(x.item()), int(y.item()), 0) for (x, y) in gt_centers]

                    # Matching detecciones | ground truth
                    TP, FP, FN = match_detections_to_gt(pred_points, gt_points,
                                                        max_dist=self.config['evaluation']['max_detection_distance'])

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                    # Loss por imagen
                    gt_density = self._prepare_density_map_batch([gt_centers], [orig_sizes_batch[i]], h_out, w_out).to(self.device)
                    loss = self.criterion(pred_density.unsqueeze(0), gt_density.unsqueeze(0))  # añadir dimensión batch
                    running_loss += loss.item()

        # Métricas globales
        precision = total_TP / (total_TP + total_FP + 1e-8)
        recall = total_TP / (total_TP + total_FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"[Validación Epoch {epoch}] Resultados globales:")
        print(f"  Loss     : {running_loss / total_batches:.4f}")
        print(f"  TP: {total_TP} | FP: {total_FP} | FN: {total_FN}")
        print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1-score: {f1:.3f}")

        return running_loss / total_batches

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
