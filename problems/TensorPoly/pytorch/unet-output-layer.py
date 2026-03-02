import torch


def unet_output(features: torch.Tensor, num_classes: int) -> torch.Tensor:
    batch, H, W, _ = features.shape
    return torch.zeros((batch, H, W, num_classes))
