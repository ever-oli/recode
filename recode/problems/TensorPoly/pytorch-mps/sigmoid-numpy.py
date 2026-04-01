import torch


def sigmoid(x, device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    return 1.0 / (1.0 + torch.exp(-x_tensor))
