import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0.0), x)


class ConvBlock:
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W1 = torch.randn(in_channels, out_channels) * 0.01
        self.W2 = torch.randn(out_channels, out_channels) * 0.01
        self.Ws = torch.randn(in_channels, out_channels) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = relu(torch.matmul(x, self.W1))
        main = torch.matmul(main, self.W2)
        shortcut = torch.matmul(x, self.Ws)
        return relu(main + shortcut)
