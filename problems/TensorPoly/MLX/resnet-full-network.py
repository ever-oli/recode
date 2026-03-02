import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)


class BasicBlock:
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.W1 = mx.random.normal(shape=(in_ch, out_ch)) * 0.01
        self.W2 = mx.random.normal(shape=(out_ch, out_ch)) * 0.01

        if in_ch != out_ch or downsample:
            self.W_proj = mx.random.normal(shape=(in_ch, out_ch)) * 0.01
        else:
            self.W_proj = None

    def forward(self, x: mx.array) -> mx.array:
        identity = x
        out = relu(mx.matmul(x, self.W1))
        out = mx.matmul(out, self.W2)

        if self.W_proj is not None:
            identity = mx.matmul(identity, self.W_proj)

        return relu(out + identity)


class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.conv1 = mx.random.normal(shape=(3, 64)) * 0.01

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

        self.fc = mx.random.normal(shape=(512, num_classes)) * 0.01

    def forward(self, x: mx.array) -> mx.array:
        out = relu(mx.matmul(x, self.conv1))

        for block in self.layer1:
            out = block.forward(out)

        for block in self.layer2:
            out = block.forward(out)

        for block in self.layer3:
            out = block.forward(out)

        for block in self.layer4:
            out = block.forward(out)

        return mx.matmul(out, self.fc)
