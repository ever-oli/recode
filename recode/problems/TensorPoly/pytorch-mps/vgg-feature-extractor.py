import torch


def conv_relu(x: torch.Tensor, out_channels: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    _, _, _, c = x.shape
    weights = torch.randn(c, out_channels, device=device) * 0.1
    x = x @ weights
    return torch.maximum(torch.tensor(0.0, device=device), x)


def maxpool_2x2(x: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(dim=2).values.max(dim=3).values


def vgg_features(x: torch.Tensor, config: list, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = x.to(device)
    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer, device=device)
        elif layer == "M":
            out = maxpool_2x2(out, device=device)
    return out
