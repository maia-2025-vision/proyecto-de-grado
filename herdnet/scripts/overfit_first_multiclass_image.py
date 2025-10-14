# Collate function para compatibilidad con el pipeline HerdNet
def custom_collate_fn(batch):
    collated = {key: [d[key] for d in batch] for key in batch[0]}
    return collated
"""
Script para probar overfitting en HerdNet usando la primera imagen con al menos 2 clases distintas y una anotación de cada una.
Lee el JSON COCO, selecciona la imagen, entrena y visualiza.
"""
import os
import json
import torch
from herdnet.data.dataset import HerdNetDataset
from herdnet.data.transforms import get_transforms
from herdnet.model.herdnet import HerdNet
from herdnet.engine.trainer import Trainer

# Configuración básica
ANNOTATION_PATH = 'data/groundtruth/json/sub_frames/train_subframes_A_B_E_K_WH_WB.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Leer JSON y buscar la primera imagen con al menos 2 clases distintas
with open(ANNOTATION_PATH, 'r') as f:
    coco = json.load(f)

# Diccionario para mostrar nombres reales en la leyenda
category_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

img_id_to_classes = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    cat_id = ann['category_id']
    img_id_to_classes.setdefault(img_id, set()).add(cat_id)

selected_img_id = None
for img_id, classes in img_id_to_classes.items():
    if len(classes) >= 1:
        # Verificar que haya al menos una anotación por clase
        counts = {c: 0 for c in classes}
        for ann in coco['annotations']:
            if ann['image_id'] == img_id:
                counts[ann['category_id']] += 1
        if all(v >= 20 for v in counts.values()):
            selected_img_id = img_id
            break

if selected_img_id is None:
    raise ValueError('No se encontró ninguna imagen con al menos 2 clases y una anotación de cada una.')

# Obtener ruta de la imagen
img_info = next(img for img in coco['images'] if img['id'] == selected_img_id)
IMAGE_PATH = os.path.join('data/train_subframes', img_info['file_name'])
IMAGE_DIR = os.path.dirname(IMAGE_PATH)
SELECTED_IMG_ID = selected_img_id
print(f"[INFO] Usando imagen: {IMAGE_PATH} con clases: {img_id_to_classes[selected_img_id]}")

# Parámetros de entrenamiento
config = {
    'training': {
        'epochs': 100,
        'batch_size': 1,
        'lr': 1e-4,
        'weight_decay': 0,
        'step_size': 20,
        'gamma': 0.5,
        'resume_from_checkpoint': None,
        'lambda_localization': 1.0,
        'lambda_classification': 1.0,
        'gaussian_sigma': 0.01
    },
    'output': {
        'checkpoint_dir': './checkpoints_overfit',
        'save_freq': 10
    },
    'dataset': {
        'num_classes': len(coco['categories'])
    },
    'evaluation': {
        'max_detection_distance': 10,
        'density_threshold': 2
    }
}

transform = get_transforms('train')
dataset = HerdNetDataset(
    annotation_path=ANNOTATION_PATH,
    image_dir=IMAGE_DIR,
    image_transforms=transform
)
# Filtrar solo la imagen seleccionada
indices = [i for i, entry in enumerate(dataset.data) if entry['image_id'] == SELECTED_IMG_ID]
assert len(indices) == 1, "No se encontró la imagen seleccionada en el dataset."
single_item = [dataset[i] for i in indices]
class SingleImageLoader(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            'image': item['image'],
            'centers': item['centers'],
            'classes': item['classes'],
            'orig_size': item['orig_size']
        }
train_loader = torch.utils.data.DataLoader(
    SingleImageLoader(single_item), batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
val_loader = torch.utils.data.DataLoader(
    SingleImageLoader(single_item), batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Modelo
model = HerdNet(backbone="resnet34", num_classes=config['dataset']['num_classes'])
model = model.to(DEVICE)

# Trainer
trainer = Trainer(model, train_loader, val_loader, config, DEVICE)
trainer.train()


# Visualización final con dual-head HerdNet
import matplotlib.pyplot as plt
print('Entrenamiento finalizado. Generando visualización de validación...')
for batch in val_loader:
    images = torch.stack(batch['image']).to(DEVICE)
    centers = batch['centers']
    classes = batch['classes']
    orig_size = batch['orig_size']
    model.eval()
    with torch.no_grad():
        loc_map, cls_map = model(images)
    from herdnet.engine.evaluator import extract_points_from_density_map, match_detections_to_gt
    # Extraer picos del mapa de localización
    print("Valor máximo en loc_map:", loc_map[0].max().item())
    pred_points_raw = extract_points_from_density_map(loc_map[0, 0].unsqueeze(0), threshold=config['evaluation']['density_threshold'])
    # Asignar clase por pixel usando el mapa de clasificación
    import torch.nn.functional as F
    cls_map_upsampled = F.interpolate(cls_map[0].unsqueeze(0), size=loc_map.shape[2:], mode='bilinear', align_corners=False).squeeze(0)
    pred_classes = torch.argmax(cls_map_upsampled, dim=0)  # [H, W]
    pred_points = [
        (int(x * images[0].shape[2] / loc_map.shape[3]), int(y * images[0].shape[1] / loc_map.shape[2]), int(pred_classes[y, x].item()))
        for (x, y, *_) in pred_points_raw
    ]
    gt_points = [(int(x.item()), int(y.item()), int(c.item())) for (x, y), c in zip(centers[0], classes[0])]

    # Matching en el espacio correcto
    TP, FP, FN = match_detections_to_gt(pred_points, gt_points, max_dist=config['evaluation']['max_detection_distance'])
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")

    # Visualización completa con 9 gráficas
    trainer.visualize_prediction(
        images[0].cpu(),
        loc_map[0].cpu(),  # Mapa de densidad con todos los canales/clases
        gt_points,
        pred_points,
        save_path='overfit_multiclass_result.png',
        category_id_to_name=category_id_to_name
    )
print('Visualización guardada en overfit_multiclass_result.png')
