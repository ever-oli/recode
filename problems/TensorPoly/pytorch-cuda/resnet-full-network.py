import torch


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)


class BasicBlock:
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.downsample = downsample
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.device = device

        self.W1 = torch.randn(in_ch, out_ch, device=device) * 0.01
        self.W2 = torch.randn(out_ch, out_ch, device=device) * 0.01

        if in_ch != out_ch or downsample:
            self.W_proj = torch.randn(in_ch, out_ch, device=device) * 0.01
        else:
            self.W_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        identity = x
        out = relu(torch.matmul(x, self.W1), device=self.device)
        out = torch.matmul(out, self.W2)

        if self.W_proj is not None:
            identity = torch.matmul(identity, self.W_proj)

        return relu(out + identity, device=self.device)


class ResNet18:
    def __init__(self, num_classes: int = 10, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.conv1 = torch.randn(3, 64, device=device) * 0.01

        self.layer1 = [
            BasicBlock(64, 64, downsample=False, device=device),
            BasicBlock(64, 64, downsample=False, device=device),
        ]

        self.layer2 = [
            BasicBlock(64, 128, downsample=True, device=device),
            BasicBlock(128, 128, downsample=False, device=device),
        ]

        self.layer3 = [
            BasicBlock(128, 256, downsample=True, device=device),
            BasicBlock(256, 256, downsample=False, device=device),
        ]

        self.layer4 = [
            BasicBlock(256, 512, downsample=True, device=device),
            BasicBlock(512, 512, downsample=False, device=device),
        ]

        self.fc = torch.randn(512, num_classes, device=device) * 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = relu(torch.matmul(x, self.conv1), device=self.device)

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
