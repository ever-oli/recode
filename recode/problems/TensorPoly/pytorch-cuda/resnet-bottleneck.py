import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)


class BottleneckBlock:
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels
        self.out_ch = out_channels
        self.device = device

        self.W1 = torch.randn(in_channels, bottleneck_channels, device=device) * 0.01
        self.W2 = torch.randn(bottleneck_channels, bottleneck_channels, device=device) * 0.01
        self.W3 = torch.randn(bottleneck_channels, out_channels, device=device) * 0.01

        self.Ws = torch.randn(in_channels, out_channels, device=device) * 0.01 if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        identity = x
        out = relu(torch.matmul(x, self.W1), device=self.device)
        out = relu(torch.matmul(out, self.W2), device=self.device)
        out = torch.matmul(out, self.W3)

        if self.Ws is not None:
            identity = torch.matmul(identity, self.Ws)

        return relu(out + identity, device=self.device)
