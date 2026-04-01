import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -500, 500)
    return 1 / (1 + torch.exp(-x))


def discriminator(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    _, input_dim = x.shape

    W1 = torch.randn(input_dim, 256, device=device) * 0.02
    b1 = torch.zeros(256, device=device)
    W2 = torch.randn(256, 128, device=device) * 0.02
    b2 = torch.zeros(128, device=device)
    W3 = torch.randn(128, 1, device=device) * 0.02
    b3 = torch.zeros(1, device=device)

    h1 = torch.matmul(x, W1) + b1
    h1 = torch.maximum(0.2 * h1, h1)
    h2 = torch.matmul(h1, W2) + b2
    h2 = torch.maximum(0.2 * h2, h2)
    logits = torch.matmul(h2, W3) + b3
    return sigmoid(logits)
