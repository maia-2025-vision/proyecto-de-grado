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
        orig_size = image.size  # (width, height)

        # Obtener centros y clases de las anotaciones
        centers = []
        classes = []
        for ann in entry['annotations']:
            x = ann['bbox'][0] + ann['bbox'][2] / 2
            y = ann['bbox'][1] + ann['bbox'][3] / 2
            centers.append([x, y])
            classes.append(ann['category_id'])

        centers = torch.tensor(centers, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        # Aplicar transformaciones si existen
        if self.transforms:
            image = self.transforms(image)

        return {
            'image': image,
            'centers': centers,
            'classes': classes,
            'orig_size': (orig_size[1], orig_size[0])  # (height, width)
        }
        
