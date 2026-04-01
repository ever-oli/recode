import torch


def conv_relu(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    _, _, _, c = x.shape
    weights = torch.randn(c, out_channels) * 0.1
    x = x @ weights
    return torch.maximum(torch.tensor(0.0), x)


def maxpool_2x2(x: torch.Tensor) -> torch.Tensor:
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(dim=2).values.max(dim=3).values


def vgg_features(x: torch.Tensor, config: list) -> torch.Tensor:
    out = x
    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer)
        elif layer == "M":
            out = maxpool_2x2(out)
    return out
