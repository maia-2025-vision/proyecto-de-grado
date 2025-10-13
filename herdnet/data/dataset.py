import os
import torch
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class HerdNetDataset(Dataset):
    def __init__(self, annotation_path, image_dir, image_transforms=None, image_size=None):
        """
        Dataset para leer datos en formato COCO y devolver imágenes + puntos centrales.
        """
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.transforms = image_transforms
        self.image_size = image_size  # sólo se aplica en entrenamiento
        self.data = self._load_coco_annotations()

    def _load_coco_annotations(self):
        """
        Carga el archivo COCO JSON y estructura los datos.
        """
        with open(self.annotation_path, 'r') as f:
            coco = json.load(f)

        # Crear diccionario: imagen_id → file_name
        image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

        # Agrupar anotaciones por imagen
        image_id_to_annots = {}
        for ann in coco['annotations']:
            if ann['iscrowd']:
                continue
            img_id = ann['image_id']
            if img_id not in image_id_to_annots:
                image_id_to_annots[img_id] = []
            image_id_to_annots[img_id].append(ann)

        # Crear lista final de ejemplos
        dataset = []
        for img_id, file_name in image_id_to_file.items():
            anns = image_id_to_annots.get(img_id, [])
            dataset.append({
                'image_id': img_id,
                'file_name': file_name,
                'annotations': anns
            })

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.image_dir, entry['file_name'])

        # Leer imagen
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size

        # Obtener puntos centrales a partir de bboxes
        centers = []
        classes = []
        for ann in entry['annotations']:
            x, y, w, h = ann['bbox']
            cx = x + w / 2
            cy = y + h / 2
            centers.append([cx, cy])
            classes.append(ann['category_id'])

        centers = np.array(centers, dtype=np.float32) if centers else np.zeros((0, 2), dtype=np.float32)
        classes = np.array(classes, dtype=np.int64) if classes else np.zeros((0,), dtype=np.int64)

        # Resize si es entrenamiento
        if self.image_size is not None:
            image = image.resize(self.image_size, Image.BILINEAR)
            scale_x = self.image_size[0] / orig_w
            scale_y = self.image_size[1] / orig_h
            centers[:, 0] *= scale_x
            centers[:, 1] *= scale_y

        # Transformaciones
        if self.transforms:
            image = self.transforms(image)

        sample = {
            'image': image,                 # Tensor [3,H,W]
            'centers': torch.tensor(centers, dtype=torch.float32),  # [N,2]
            'classes': torch.tensor(classes, dtype=torch.long),     # [N]
            'image_id': entry['image_id'],
            'orig_size': (orig_w, orig_h)
        }

        return sample
