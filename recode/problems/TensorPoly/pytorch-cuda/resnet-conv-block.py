import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)


class ConvBlock:
    def __init__(self, in_channels: int, out_channels: int, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.W1 = torch.randn(in_channels, out_channels, device=device) * 0.01
        self.W2 = torch.randn(out_channels, out_channels, device=device) * 0.01
        self.Ws = torch.randn(in_channels, out_channels, device=device) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        main = relu(torch.matmul(x, self.W1), device=self.device)
        main = torch.matmul(main, self.W2)
        shortcut = torch.matmul(x, self.Ws)
        return relu(main + shortcut, device=self.device)
