import torch


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
