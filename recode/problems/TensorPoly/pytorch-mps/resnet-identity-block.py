import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)


class IdentityBlock:
    def __init__(self, channels: int, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.channels = channels
        self.device = device
        self.W1 = torch.randn(channels, channels, device=device) * 0.01
        self.W2 = torch.randn(channels, channels, device=device) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        identity = x
        out = relu(torch.matmul(x, self.W1), device=self.device)
        out = torch.matmul(out, self.W2)
        return out + identity
