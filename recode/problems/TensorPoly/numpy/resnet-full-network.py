import numpy as np


def relu(x):
    return np.maximum(0, x)


class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01

        if in_ch != out_ch or downsample:
            self.W_proj = np.random.randn(in_ch, out_ch) * 0.01
        else:
            self.W_proj = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        identity = x
        out = relu(np.matmul(x, self.W1))
        out = np.matmul(out, self.W2)

        if self.W_proj is not None:
            identity = np.matmul(identity, self.W_proj)

        out = relu(out + identity)
        return out


class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01

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

        self.fc = np.random.randn(512, num_classes) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = relu(np.matmul(x, self.conv1))

        for block in self.layer1:
            out = block.forward(out)

        for block in self.layer2:
            out = block.forward(out)

        for block in self.layer3:
            out = block.forward(out)

        for block in self.layer4:
            out = block.forward(out)

        logits = np.matmul(out, self.fc)
        return logits
