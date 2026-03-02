import torch


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.tensor(0.0), x)


class BasicBlock:
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.W1 = torch.randn(in_ch, out_ch) * 0.01
        self.W2 = torch.randn(out_ch, out_ch) * 0.01

        if in_ch != out_ch or downsample:
            self.W_proj = torch.randn(in_ch, out_ch) * 0.01
        else:
            self.W_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = relu(torch.matmul(x, self.W1))
        out = torch.matmul(out, self.W2)

        if self.W_proj is not None:
            identity = torch.matmul(identity, self.W_proj)

        return relu(out + identity)


class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.conv1 = torch.randn(3, 64) * 0.01

        self.layer1 = [
            BasicBlock(64, 64, downsample=False),
            BasicBlock(64, 64, downsample=False),
        ]

        self.layer2 = [
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128, downsample=False),
        ]

        self.layer3 = [
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256, downsample=False),
        ]

        self.layer4 = [
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512, downsample=False),
        ]

        self.fc = torch.randn(512, num_classes) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(torch.matmul(x, self.conv1))

        for block in self.layer1:
            out = block.forward(out)

        for block in self.layer2:
            out = block.forward(out)

        for block in self.layer3:
            out = block.forward(out)

        for block in self.layer4:
            out = block.forward(out)

        logits = torch.matmul(out, self.fc)
        return logits
