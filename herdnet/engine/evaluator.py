import torch
import numpy as np
import scipy.ndimage
from sklearn.metrics import precision_score, recall_score, f1_score

def extract_points_from_density_map(density_map, threshold=0.1, min_distance=3):
    """
    Extrae los picos (puntos) del mapa de densidad usando supresión no máxima.
    Retorna una lista de puntos por clase.
    """
    density_map = density_map.cpu().numpy()  # [C, H, W]
    detections = []

    for c in range(density_map.shape[0]):
        class_map = density_map[c]

        #aplicar suavizado (puede mejorar la detección de picos)
        class_map = scipy.ndimage.gaussian_filter(class_map, sigma=1)

        # Buscar picos locales por encima del umbral
        mask = (class_map > threshold)
        peaks = scipy.ndimage.maximum_filter(class_map, size=min_distance) == class_map
        peaks = peaks & mask

        ys, xs = np.where(peaks)
        class_points = [(int(x), int(y), c) for x, y in zip(xs, ys)]

        detections.extend(class_points)

    return detections  # [(x, y, class_id), ...]

def match_detections_to_gt(pred_points, gt_points, max_dist=10):
    """
    Matchea detecciones predichas con GT basados en distancia.
    """
    matched = []
    unmatched_preds = []
    unmatched_gts = set(range(len(gt_points)))

    used_preds = set()

    print("\n[Matching Log]")
    for i, pred in enumerate(pred_points):
        px, py, pc = pred
        best_gt = None
        best_dist = max_dist + 1
        print(f"Pred[{i}] ({px},{py}, clase={pc}):")
        for j, gt in enumerate(gt_points):
            if j not in unmatched_gts:
                continue
            gx, gy, gc = gt
            if pc != gc:
                continue  # solo comparamos dentro de la misma clase
            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
            # print(f"  vs GT[{j}] ({gx},{gy}, clase={gc}) -> dist={dist:.2f}")
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_gt = j
        if best_gt is not None:
            print(f"  -> MATCH con GT[{best_gt}] (dist={best_dist:.2f})")
            matched.append((i, best_gt))
            unmatched_gts.remove(best_gt)
            used_preds.add(i)
        else:
            print("  -> NO MATCH")
            unmatched_preds.append(i)

    TP = len(matched)
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    return TP, FP, FN
