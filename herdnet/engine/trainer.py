import os
import time
from herdnet.engine.evaluator import extract_points_from_density_map, match_detections_to_gt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
from tqdm import tqdm

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

        total_batches = len(self.train_loader)
        print(f"[Entrenamiento] Epoch {epoch} - Total batches: {total_batches}")
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Entrenamiento")):
            images = torch.stack(batch['image']).to(self.device)  # [B, 3, H, W]
            centers = batch['centers']
            classes = batch['classes']
            orig_sizes = batch['orig_size']

            bs, _, h_in, w_in = images.shape

            outputs = self.model(images)  # [B, C, H_out, W_out]
            _, _, h_out, w_out = outputs.shape

            # Generar ground truth density maps al tamaño de salida
            gt_density = self._prepare_density_map_batch(list(zip(centers, classes)), orig_sizes, h_out, w_out).to(self.device)

            self.optimizer.zero_grad()
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

        total_batches = len(self.val_loader)
        print(f"[Validación] Epoch {epoch} - Total batches: {total_batches}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validación")):
                images = torch.stack(batch['image']).to(self.device)   # [B, 3, H, W]
                centers_batch = batch['centers']
                classes_batch = batch['classes']
                orig_sizes_batch = batch['orig_size']
                bs, _, h_in, w_in = images.shape

                outputs = self.model(images)  # [B, C, H_out, W_out]
                _, _, h_out, w_out = outputs.shape

                density_threshold = self.config['evaluation'].get('density_threshold', 0.5)
                for i in range(bs):
                    pred_density = outputs[i]  # [C, H_out, W_out]
                    pred_points = extract_points_from_density_map(pred_density, threshold=density_threshold)

                    gt_centers = centers_batch[i]  # Tensor [N, 2]
                    gt_classes = classes_batch[i]
                    gt_points = [(int(x.item()), int(y.item()), int(c.item()))
                                 for (x, y), c in zip(gt_centers, gt_classes)]

                    # Matching detecciones | ground truth
                    TP, FP, FN = match_detections_to_gt(pred_points, gt_points,
                                                        max_dist=self.config['evaluation']['max_detection_distance'])

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                    # Visualización rápida solo para la primera imagen del primer batch
                    if batch_idx == 0 and i == 0:
                        # Recuperar imagen original (sin normalizar)
                        img_vis = batch['image'][i]
                        # self.visualize_prediction(img_vis, pred_density, gt_points, pred_points)

                    # Loss por imagen
                    gt_density = self._prepare_density_map_batch([(gt_centers, gt_classes)], [orig_sizes_batch[i]], h_out, w_out).to(self.device)
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
        ubicando una gaussiana en el canal de la clase correspondiente a cada centro.
        """
        sigma = self.config['training'].get('gaussian_sigma', 3)
        kernel_size = int(6 * sigma)
        maps = []
        for centers, classes in batch_centers:
            density = torch.zeros((self.config['dataset']['num_classes'], height, width))
            for idx, (xy, class_id) in enumerate(zip(centers, classes)):
                x = int(round(xy[0].item()))
                y = int(round(xy[1].item()))
                c = int(class_id.item())
                if 0 <= x < width and 0 <= y < height and 0 <= c < density.shape[0]:
                    x_min = max(0, x - kernel_size)
                    x_max = min(width, x + kernel_size + 1)
                    y_min = max(0, y - kernel_size)
                    y_max = min(height, y + kernel_size + 1)
                    yy, xx = torch.meshgrid(
                        torch.arange(y_min, y_max).float(),
                        torch.arange(x_min, x_max).float(),
                        indexing='ij'
                    )
                    gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                    density[c, y_min:y_max, x_min:x_max] += gaussian
            maps.append(density)

        return torch.stack(maps)  # [B, C, H, W]

    def _prepare_density_map_batch(self, centers_list, orig_sizes, h_out, w_out):
        """
        Escala los centros de cada imagen del batch a la resolución de salida y genera el mapa de densidad.
        """
        scaled_centers_and_classes = []
        for i, (centers, classes) in enumerate(centers_list):
            orig_w, orig_h = orig_sizes[i]
            scale_x = w_out / orig_w
            scale_y = h_out / orig_h
            centers_i = centers.clone()
            if centers_i.numel() > 0:
                centers_i[:, 0] *= scale_x
                centers_i[:, 1] *= scale_y
            scaled_centers_and_classes.append((centers_i, classes))
        return self._generate_density_map(scaled_centers_and_classes, h_out, w_out)

    def _save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, ckpt_path)
        print(f"[INFO] Checkpoint guardado: {ckpt_path}")
    
    def visualize_prediction(self, image, pred_density, gt_points, pred_points, save_path=None, category_id_to_name=None, img_path=None):
        """
        Visualiza la imagen con los puntos GT (verde), predicciones (rojo) y heatmap de densidad por clase.
        Además imprime las clases y coordenadas de los puntos GT y predichos.
        """
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt
        from herdnet.data.transforms import unnormalize_image

        # Imprimir puntos GT
        print("\n--- Ground Truth ---")
        for idx, (x, y, c) in enumerate(gt_points):
            nombre = category_id_to_name.get(c, str(c)) if category_id_to_name else str(c)
            print(f"GT[{idx}]: clase={nombre}, (x={x}, y={y})")

        # Desnormalizar imagen antes de visualizar usando la función utilitaria
        img_np = unnormalize_image(image)
        h_img, w_img = img_np.shape[:2]
        h_map, w_map = pred_density.shape[1:]

        # Imprimir puntos predichos (coordenadas reescaladas a la imagen original)
        print("\n--- Predicciones ---")
        for idx, (x, y, c) in enumerate(pred_points):
            nombre = category_id_to_name.get(c, str(c)) if category_id_to_name else str(c)
            x_img = int(x * w_img / w_map)
            y_img = int(y * h_img / h_map)
            print(f"Pred[{idx}]: clase={nombre}, (x={x_img}, y={y_img})")

        # Desnormalizar imagen antes de visualizar usando la función utilitaria
        img_np = unnormalize_image(image)

        # Imagen con puntos GT y predicciones
        img_vis = img_np.copy()
        h_img, w_img = img_np.shape[:2]
        h_map, w_map = pred_density.shape[1:]
        # Colores sugeridos por clase
        colores = {
            0: (255, 0, 0),     # azul
            1: (255, 140, 0),  # naranja
            2: (0, 0, 255),    # rojo
            3: (255, 255, 0),  # cyan
            4: (255, 0, 255),  # magenta
            5: (0, 255, 255),  # amarillo
        }
        radio_punto = 8
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        # Dibujar puntos GT (siempre verde), sin texto
        for (x, y, c) in gt_points:
            color = (0, 255, 0)
            cv2.circle(img_vis, (int(x), int(y)), radius=radio_punto, color=color, thickness=-1)
        # Reescalar puntos predichos al espacio de la imagen original (color por clase, nombre)
        for (x, y, c) in pred_points:
            x_img = int(x * w_img / w_map)
            y_img = int(y * h_img / h_map)
            color = colores.get(c, (0, 0, 255))
            cv2.circle(img_vis, (x_img, y_img), radius=radio_punto, color=color, thickness=-1)

        num_classes = pred_density.shape[0]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        # Mostrar el path de la imagen original como título principal si se pasa como argumento
        if img_path:
            fig.suptitle(f'Imagen evaluada: {img_path}', fontsize=14)

        # Primer fila: [GT sobre imagen], [Predicciones sobre imagen], [Leyenda]
        # GT sobre imagen original
        img_gt = img_np.copy()
        for (x, y, c) in gt_points:
            cv2.circle(img_gt, (int(x), int(y)), radius=radio_punto, color=(0, 255, 0), thickness=-1)
        axes[0, 0].imshow(img_gt)
        axes[0, 0].set_title('GT: verde sobre imagen')
        axes[0, 0].axis('off')

        # Predicciones sobre imagen original
        img_pred = img_np.copy()
        for (x, y, c) in pred_points:
            color = colores.get(c, (0, 0, 255))
            cv2.circle(img_pred, (int(x), int(y)), radius=radio_punto, color=color, thickness=-1)
        axes[0, 1].imshow(img_pred)
        axes[0, 1].set_title('Predicciones sobre imagen')
        axes[0, 1].axis('off')

        # Leyenda en la celda [0,2]
        axes[0, 2].axis('off')
        import matplotlib.patches as mpatches
        legend_patches = []
        clases_predichas = set([c for (_, _, c) in pred_points])
        for class_id in clases_predichas:
            color = colores.get(class_id, (0, 0, 255))
            if category_id_to_name:
                nombre = category_id_to_name.get(class_id, str(class_id))
            else:
                nombre = str(class_id)
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
                # Obtener nombre de la clase si existe
                if category_id_to_name:
                    nombre = category_id_to_name.get(i, str(i))
                else:
                    nombre = str(i)
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
