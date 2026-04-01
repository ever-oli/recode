import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0.0), x)


class IdentityBlock:
    def __init__(self, channels: int):
        self.channels = channels
        self.W1 = torch.randn(channels, channels) * 0.01
        self.W2 = torch.randn(channels, channels) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = relu(torch.matmul(x, self.W1))
        out = torch.matmul(out, self.W2)
        return out + identity
