import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def unnormalize_image(img_tensor):
    """
    Desnormaliza una imagen normalizada con los valores de ImageNet.
    img_tensor: torch.Tensor [C, H, W] o [B, C, H, W]
    Devuelve: numpy array [H, W, C] en uint8
    """
    imagenet_mean = np.array(IMAGENET_MEAN)
    imagenet_std = np.array(IMAGENET_STD)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]  # tomar la primera si es batch
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * imagenet_std + imagenet_mean)
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def get_transforms(mode='train', image_size=(2000, 2000)):
    """
    Devuelve las transformaciones de imagen para entrenamiento o validación.
    """
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])
    elif mode == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Modo de transformaciones no válido: {mode}")

    return transform
