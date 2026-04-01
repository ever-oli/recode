import torch


def unet_output(features: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    features = features.to(device)
    batch, H, W, _ = features.shape
    return torch.zeros((batch, H, W, num_classes), device=device)
