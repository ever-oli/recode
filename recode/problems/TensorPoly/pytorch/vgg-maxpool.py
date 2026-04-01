import torch


def vgg_maxpool(x: torch.Tensor) -> torch.Tensor:
    batch, h, w, c = x.shape
    reshaped_x = x.reshape(batch, h // 2, 2, w // 2, 2, c)
    return reshaped_x.max(dim=2).values.max(dim=3).values
