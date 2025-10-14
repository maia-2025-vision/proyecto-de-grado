import json
from collections import defaultdict

# Ruta al archivo COCO JSON
json_path = "data/groundtruth/json/sub_frames/train_subframes_A_B_E_K_WH_WB.json"

def main():
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Agrupar clases por imagen
    image_classes = defaultdict(set)
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        class_id = ann['category_id']
        image_classes[img_id].add(class_id)

    # Buscar imágenes con más de una clase
    multi_class_images = [img_id for img_id, classes in image_classes.items() if len(classes) > 1]

    print(f"Total de imágenes con más de una clase: {len(multi_class_images)}")
    if multi_class_images:
        print("IDs de imágenes con múltiples clases:")
        for img_id in multi_class_images:
            print(f"Imagen ID: {img_id}, Clases: {sorted(image_classes[img_id])}")
    else:
        print("No se encontraron imágenes con más de una clase.")

if __name__ == "__main__":
    main()
