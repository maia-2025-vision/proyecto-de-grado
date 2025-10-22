import torch


def pick_torch_device() -> str:
    """Pick best accelerator device for current machine, or default to cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"
