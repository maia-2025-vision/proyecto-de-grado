from torchvision import transforms

def get_transforms(mode='train', image_size=(2000, 2000)):
    """
    Devuelve las transformaciones de imagen para entrenamiento o validación.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImagenNet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
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
