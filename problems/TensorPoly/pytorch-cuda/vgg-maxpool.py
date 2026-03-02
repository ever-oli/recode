import torch


def vgg_maxpool(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    batch, h, w, c = x.shape
    reshaped_x = x.reshape(batch, h // 2, 2, w // 2, 2, c)
    return reshaped_x.max(dim=2).values.max(dim=3).values
