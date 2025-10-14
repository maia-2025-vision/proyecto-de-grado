import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from herdnet.engine.evaluator import extract_points_from_density_map, match_detections_to_gt


class Trainer:
    def visualize_prediction(self, image, pred_density, gt_points, pred_points, save_path=None, category_id_to_name=None, img_path=None):
        """
        Visualiza la imagen con los puntos GT (verde), predicciones (color por clase) y heatmap de densidad por clase.
        Imprime las clases y coordenadas de los puntos GT y predichos.
        """
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        try:
            from herdnet.data.transforms import unnormalize_image
            img_np = unnormalize_image(image)
        except Exception:
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        h_img, w_img = img_np.shape[:2]
        h_map, w_map = pred_density.shape[1:]

        # Imprimir puntos GT
        print("\n--- Ground Truth ---")
        for idx, (x, y, c) in enumerate(gt_points):
            nombre = category_id_to_name.get(c, str(c)) if category_id_to_name else str(c)
            print(f"GT[{idx}]: clase={nombre}, (x={x}, y={y})")

        # Imprimir puntos predichos
        print("\n--- Predicciones ---")
        for idx, (x, y, c) in enumerate(pred_points):
            nombre = category_id_to_name.get(c, str(c)) if category_id_to_name else str(c)
            x_img = int(x * w_img / w_map)
            y_img = int(y * h_img / h_map)
            print(f"Pred[{idx}]: clase={nombre}, (x={x_img}, y={y_img})")

        # Imagen con puntos GT y predicciones
        img_vis = img_np.copy()
        colores = {
            0: (255, 0, 0),     # azul
            1: (255, 140, 0),  # naranja
            2: (0, 0, 255),    # rojo
            3: (255, 255, 0),  # cyan
            4: (255, 0, 255),  # magenta
            5: (0, 255, 255),  # amarillo
        }
        radio_punto = 8
        # Dibujar puntos GT (verde)
        for (x, y, c) in gt_points:
            cv2.circle(img_vis, (int(x), int(y)), radius=radio_punto, color=(0, 255, 0), thickness=-1)
        # Dibujar puntos predichos (color por clase)
        for (x, y, c) in pred_points:
            x_img = int(x * w_img / w_map)
            y_img = int(y * h_img / h_map)
            color = colores.get(c, (0, 0, 255))
            cv2.circle(img_vis, (x_img, y_img), radius=radio_punto, color=color, thickness=-1)

        num_classes = pred_density.shape[0]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        if img_path:
            fig.suptitle(f'Imagen evaluada: {img_path}', fontsize=14)

        # Primer fila: GT, predicciones, leyenda
        img_gt = img_np.copy()
        for (x, y, c) in gt_points:
            cv2.circle(img_gt, (int(x), int(y)), radius=radio_punto, color=(0, 255, 0), thickness=-1)
        axes[0, 0].imshow(img_gt)
        axes[0, 0].set_title('GT: verde sobre imagen')
        axes[0, 0].axis('off')

        img_pred = img_np.copy()
        for (x, y, c) in pred_points:
            color = colores.get(c, (0, 0, 255))
            cv2.circle(img_pred, (int(x), int(y)), radius=radio_punto, color=color, thickness=-1)
        axes[0, 1].imshow(img_pred)
        axes[0, 1].set_title('Predicciones sobre imagen')
        axes[0, 1].axis('off')

        axes[0, 2].axis('off')
        import matplotlib.patches as mpatches
        legend_patches = []
        clases_predichas = set([c for (_, _, c) in pred_points])
        for class_id in clases_predichas:
            color = colores.get(class_id, (0, 0, 255))
            nombre = category_id_to_name.get(class_id, str(class_id)) if category_id_to_name else str(class_id)
            patch = mpatches.Patch(color=np.array(color)/255.0, label=nombre)
            legend_patches.append(patch)
        axes[0, 2].legend(handles=legend_patches, loc='center', fontsize=12, frameon=False)
        axes[0, 2].set_title('Leyenda clases predichas')

        # Segunda y tercera fila: heatmaps de las 6 clases
        for i in range(6):
            row = 1 if i < 3 else 2
            col = i if i < 3 else i - 3
            if num_classes > i:
                heatmap = pred_density[i].cpu().numpy()
                nombre = category_id_to_name.get(i, str(i)) if category_id_to_name else str(i)
                axes[row, col].imshow(heatmap, cmap='hot')
                axes[row, col].set_title(f'Heatmap: {nombre}')
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')
        for r in range(3):
            for c in range(3):
                if not axes[r, c].has_data():
                    axes[r, c].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.num_classes = config['dataset']['num_classes']

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

        # Losses
        self.localization_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Pesos de los losses
        self.lambda_loc = config['training'].get('lambda_localization', 1.0)
        self.lambda_cls = config['training'].get('lambda_classification', 1.0)

        # Checkpoints
        self.checkpoint_dir = config['output']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Cargar desde checkpoint si aplica
        ckpt_path = config['training'].get('resume_from_checkpoint')
        if ckpt_path:
            self._load_checkpoint(ckpt_path)

    def train(self):
        print(f"[INFO] Iniciando entrenamiento por {self.epochs} epochs.")
        print(f"[INFO] Lambda Loc: {self.lambda_loc}, Lambda Cls: {self.lambda_cls}")
        
        best_f1 = 0.0

        for epoch in range(1, self.epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{self.epochs}")
            print('='*70)

            train_loss, train_loc_loss, train_cls_loss = self._train_one_epoch(epoch)
            val_loss, val_loc_loss, val_cls_loss, metrics = self._validate(epoch)

            print(f"\n[Resumen Epoch {epoch}]")
            print(f"  Train - Total: {train_loss:.4f} | Loc: {train_loc_loss:.4f} | Cls: {train_cls_loss:.4f}")
            print(f"  Val   - Total: {val_loss:.4f} | Loc: {val_loc_loss:.4f} | Cls: {val_cls_loss:.4f}")
            print(f"  Métricas - P: {metrics['precision']:.3f} | R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f}")

            self.scheduler.step()

            # Guardar checkpoint periódicamente
            # if epoch % self.config['output']['save_freq'] == 0:
                # self._save_checkpoint(epoch, metrics)

            # Guardar mejor modelo
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                # self._save_best_model(epoch, metrics)
                print(f"  ⭐ Nuevo mejor modelo! F1={best_f1:.3f}")

    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_loc_loss = 0.0
        running_cls_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = torch.stack(batch['image']).to(self.device)
            centers = batch['centers']
            classes = batch['classes']
            orig_sizes = batch.get('orig_size', [(img.shape[2], img.shape[1]) for img in batch['image']])

            bs = images.shape[0]

            # Forward
            loc_map_pred, cls_map_pred = self.model(images)
            # loc_map_pred: [B, 1, H_loc, W_loc] - alta resolución
            # cls_map_pred: [B, num_classes, H_cls, W_cls] - baja resolución

            _, _, h_loc, w_loc = loc_map_pred.shape
            _, _, h_cls, w_cls = cls_map_pred.shape

            # Generar GT para cada cabeza
            gt_loc_map = self._generate_localization_gt(
                centers, classes, orig_sizes, h_loc, w_loc
            ).to(self.device)
            
            gt_cls_map = self._generate_classification_gt(
                centers, classes, orig_sizes, h_cls, w_cls
            ).to(self.device)

            # Calcular losses
            loss_loc = self.localization_criterion(loc_map_pred, gt_loc_map)
            
            # Classification loss solo donde hay objetos
            mask = (gt_loc_map > 0.1).float()  # [B, 1, H, W]
            
            # Interpolar mask a resolución de classification
            if mask.shape[2:] != (h_cls, w_cls):
                mask_cls = F.interpolate(mask, size=(h_cls, w_cls), mode='nearest')
            else:
                mask_cls = mask
            
            loss_cls_map = self.classification_criterion(cls_map_pred, gt_cls_map.long())  # [B, H, W]
            loss_cls = (loss_cls_map * mask_cls.squeeze(1)).sum() / (mask_cls.sum() + 1e-8)

            # Loss total
            total_loss = self.lambda_loc * loss_loc + self.lambda_cls * loss_cls

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Acumular
            running_loss += total_loss.item()
            running_loc_loss += loss_loc.item()
            running_cls_loss += loss_cls.item()

            # Actualizar barra
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'loc': f'{loss_loc.item():.4f}',
                'cls': f'{loss_cls.item():.4f}'
            })

        n = len(self.train_loader)
        return running_loss / n, running_loc_loss / n, running_cls_loss / n

    def _validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_loc_loss = 0.0
        running_cls_loss = 0.0

        total_TP = 0
        total_FP = 0
        total_FN = 0

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = torch.stack(batch['image']).to(self.device)
                centers = batch['centers']
                classes = batch['classes']
                orig_sizes = batch.get('orig_size', [(img.shape[2], img.shape[1]) for img in batch['image']])

                bs = images.shape[0]

                # Forward
                loc_map_pred, cls_map_pred = self.model(images)
                _, _, h_loc, w_loc = loc_map_pred.shape
                _, _, h_cls, w_cls = cls_map_pred.shape

                # GT
                gt_loc_map = self._generate_localization_gt(
                    centers, classes, orig_sizes, h_loc, w_loc
                ).to(self.device)
                
                gt_cls_map = self._generate_classification_gt(
                    centers, classes, orig_sizes, h_cls, w_cls
                ).to(self.device)

                # Losses
                loss_loc = self.localization_criterion(loc_map_pred, gt_loc_map)
                
                mask = (gt_loc_map > 0.1).float()
                if mask.shape[2:] != (h_cls, w_cls):
                    mask_cls = F.interpolate(mask, size=(h_cls, w_cls), mode='nearest')
                else:
                    mask_cls = mask
                
                loss_cls_map = self.classification_criterion(cls_map_pred, gt_cls_map.long())
                loss_cls = (loss_cls_map * mask_cls.squeeze(1)).sum() / (mask_cls.sum() + 1e-8)

                total_loss = self.lambda_loc * loss_loc + self.lambda_cls * loss_cls

                running_loss += total_loss.item()
                running_loc_loss += loss_loc.item()
                running_cls_loss += loss_cls.item()

                # Métricas: extraer detecciones
                density_threshold = self.config['evaluation'].get('density_threshold', 0.3)
                
                for i in range(bs):
                    # ✅ CORRECTO: mantener dimensión de canal
                    pred_loc_map = loc_map_pred[i]  # [1, H, W]
                    pred_cls_map = cls_map_pred[i]  # [num_classes, H, W]

                    # Extraer puntos del mapa de localización
                    pred_points = extract_points_from_density_map(
                        pred_loc_map, 
                        threshold=density_threshold
                    )

                    # Interpolar classification a resolución de localization
                    pred_cls_upsampled = F.interpolate(
                        pred_cls_map.unsqueeze(0),
                        size=(h_loc, w_loc),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [num_classes, H_loc, W_loc]

                    # Asignar clase a cada punto
                    pred_points_with_class = []
                    for x, y, _ in pred_points:
                        if 0 <= y < h_loc and 0 <= x < w_loc:
                            class_id = torch.argmax(pred_cls_upsampled[:, y, x]).item()
                            pred_points_with_class.append((x, y, class_id))

                    # GT points (escalar a resolución de localization)
                    gt_centers = centers[i]
                    gt_classes = classes[i]
                    orig_w, orig_h = orig_sizes[i]
                    
                    gt_points = []
                    for (cx, cy), cls in zip(gt_centers, gt_classes):
                        x_scaled = int(round(cx.item() * w_loc / orig_w))
                        y_scaled = int(round(cy.item() * h_loc / orig_h))
                        gt_points.append((x_scaled, y_scaled, int(cls.item())))

                    # Matching
                    TP, FP, FN = match_detections_to_gt(
                        pred_points_with_class,
                        gt_points,
                        max_dist=self.config['evaluation']['max_detection_distance']
                    )

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        # Métricas globales
        n = len(self.val_loader)
        avg_loss = running_loss / n
        avg_loc_loss = running_loc_loss / n
        avg_cls_loss = running_cls_loss / n

        precision = total_TP / (total_TP + total_FP + 1e-8)
        recall = total_TP / (total_TP + total_FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        print(f"\n[Métricas Validación]")
        print(f"  TP: {total_TP} | FP: {total_FP} | FN: {total_FN}")

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'TP': total_TP,
            'FP': total_FP,
            'FN': total_FN
        }

        return avg_loss, avg_loc_loss, avg_cls_loss, metrics

    def _generate_localization_gt(self, centers_batch, classes_batch, orig_sizes, h_out, w_out):
        """
        Genera mapa de localización GT con gaussianas.
        Output: [B, 1, h_out, w_out]
        """
        batch_size = len(centers_batch)
        gt_map = torch.zeros(batch_size, 1, h_out, w_out)
        
        sigma = self.config['training'].get('gaussian_sigma', 3)
        kernel_size = int(6 * sigma)

        for b in range(batch_size):
            centers = centers_batch[b]
            orig_w, orig_h = orig_sizes[b]
            
            for cx, cy in centers:
                # Escalar a resolución de salida
                x = int(round(cx.item() * w_out / orig_w))
                y = int(round(cy.item() * h_out / orig_h))
                
                # Gaussiana local
                x_min = max(0, x - kernel_size)
                x_max = min(w_out, x + kernel_size + 1)
                y_min = max(0, y - kernel_size)
                y_max = min(h_out, y + kernel_size + 1)
                
                if x_min < x_max and y_min < y_max:
                    yy, xx = torch.meshgrid(
                        torch.arange(y_min, y_max).float(),
                        torch.arange(x_min, x_max).float(),
                        indexing='ij'
                    )
                    
                    gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                    gt_map[b, 0, y_min:y_max, x_min:x_max] += gaussian

        return gt_map

    def _generate_classification_gt(self, centers_batch, classes_batch, orig_sizes, h_out, w_out):
        """
        Genera mapa de clasificación GT (índice de clase por píxel).
        Output: [B, h_out, w_out] con índices de clase
        """
        batch_size = len(centers_batch)
        gt_map = torch.zeros(batch_size, h_out, w_out, dtype=torch.long)

        for b in range(batch_size):
            centers = centers_batch[b]
            classes = classes_batch[b]
            orig_w, orig_h = orig_sizes[b]
            
            for (cx, cy), cls in zip(centers, classes):
                x = int(round(cx.item() * w_out / orig_w))
                y = int(round(cy.item() * h_out / orig_h))
                
                x = max(0, min(w_out - 1, x))
                y = max(0, min(h_out - 1, y))
                
                gt_map[b, y, x] = int(cls.item())

        return gt_map

    def _save_checkpoint(self, epoch, metrics=None):
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, ckpt_path)
        print(f"[INFO] Checkpoint guardado: {ckpt_path}")

    def _save_best_model(self, epoch, metrics):
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, best_path)

    def _load_checkpoint(self, ckpt_path):
        print(f"[INFO] Cargando checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"[INFO] Checkpoint cargado desde epoch {checkpoint['epoch']}")