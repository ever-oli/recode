import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0.0), x)


class BottleneckBlock:
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels
        self.out_ch = out_channels

        self.W1 = torch.randn(in_channels, bottleneck_channels) * 0.01
        self.W2 = torch.randn(bottleneck_channels, bottleneck_channels) * 0.01
        self.W3 = torch.randn(bottleneck_channels, out_channels) * 0.01

        self.Ws = torch.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = relu(torch.matmul(x, self.W1))
        out = relu(torch.matmul(out, self.W2))
        out = torch.matmul(out, self.W3)

        if self.Ws is not None:
            identity = torch.matmul(identity, self.Ws)

        return relu(out + identity)
