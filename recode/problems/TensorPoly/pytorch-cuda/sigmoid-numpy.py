import torch


def sigmoid(x, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return 1.0 / (1.0 + torch.exp(-x_tensor))
