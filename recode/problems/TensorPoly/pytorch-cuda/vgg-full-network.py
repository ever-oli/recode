import torch


def vgg16(x: torch.Tensor, num_classes: int = 1000, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vgg16_config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]

    features = vgg_features(x.to(device), vgg16_config, device=device)
    return vgg_classifier(features, num_classes, device=device)


def conv_relu(x: torch.Tensor, out_channels: int, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    _, _, _, c = x.shape
    weights = torch.randn(c, out_channels, device=device) * 0.1
    x = x @ weights
    return torch.maximum(torch.tensor(0.0, device=device), x)


def maxpool_2x2(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    b, h, w, c = x.shape
    return x.reshape(b, h // 2, 2, w // 2, 2, c).max(dim=2).values.max(dim=3).values


def vgg_features(x: torch.Tensor, config: list, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = x.to(device)
    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer, device=device)
        elif layer == "M":
            out = maxpool_2x2(out, device=device)
    return out


def vgg_classifier(features: torch.Tensor, num_classes: int = 1000, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    batch_size = features.shape[0]
    x = features.reshape(batch_size, -1)

    def dense_relu(input_data: torch.Tensor, out_dim: int) -> torch.Tensor:
        in_dim = input_data.shape[1]
        limit = torch.sqrt(torch.tensor(2.0 / in_dim, device=device))
        w = torch.randn(in_dim, out_dim, device=device) * limit
        b = torch.zeros(out_dim, device=device)
        return torch.maximum(torch.tensor(0.0, device=device), input_data @ w + b)

    x = dense_relu(x, 4096)
    x = dense_relu(x, 4096)

    in_dim_final = x.shape[1]
    w_final = torch.randn(in_dim_final, num_classes, device=device) * torch.sqrt(torch.tensor(2.0 / in_dim_final, device=device))
    b_final = torch.zeros(num_classes, device=device)
    return x @ w_final + b_final
