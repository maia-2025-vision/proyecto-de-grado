import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_detections(image, gt_points, pred_points, figsize=(10, 10), save_path=None):
    """
    Muestra o guarda una imagen con puntos GT (verde) y predicciones (rojo).
    
    - image: tensor [3, H, W] o array [H, W, 3]
    - gt_points: lista de (x, y, clase)
    - pred_points: lista de (x, y, clase)
    """
    if isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        img_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img_np = (img_np * 255).astype(np.uint8)

    img_vis = img_np.copy()

    # Dibujar puntos GT en verde
    for (x, y, _) in gt_points:
        cv2.circle(img_vis, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

    # Dibujar puntos predichos en rojo
    for (x, y, _) in pred_points:
        cv2.circle(img_vis, (x, y), radius=4, color=(255, 0, 0), thickness=1)

    # Mostrar o guardar
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Imagen guardada en {save_path}")
    else:
        plt.show()
