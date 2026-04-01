import torch


def vgg16(x: torch.Tensor, num_classes: int = 1000) -> torch.Tensor:
    vgg16_config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]

    features = vgg_features(x, vgg16_config)
    return vgg_classifier(features, num_classes)


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


def vgg_classifier(features: torch.Tensor, num_classes: int = 1000) -> torch.Tensor:
    batch_size = features.shape[0]
    x = features.reshape(batch_size, -1)

    def dense_relu(input_data: torch.Tensor, out_dim: int) -> torch.Tensor:
        in_dim = input_data.shape[1]
        limit = torch.sqrt(torch.tensor(2.0 / in_dim))
        w = torch.randn(in_dim, out_dim) * limit
        b = torch.zeros(out_dim)
        return torch.maximum(torch.tensor(0.0), input_data @ w + b)

    x = dense_relu(x, 4096)
    x = dense_relu(x, 4096)

    in_dim_final = x.shape[1]
    w_final = torch.randn(in_dim_final, num_classes) * torch.sqrt(torch.tensor(2.0 / in_dim_final))
    b_final = torch.zeros(num_classes)
    return x @ w_final + b_final
