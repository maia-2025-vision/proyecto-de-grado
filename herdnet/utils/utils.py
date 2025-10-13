import os
import random
import numpy as np
import torch

def set_seed(seed):
    """
    Establece la semilla para reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Para que los resultados sean más reproducibles, aunque puede afectar el rendimiento
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_device(device_str):
    """
    Configura el dispositivo (GPU o CPU).
    """
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[INFO] Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"[INFO] Usando CPU")
    return device
