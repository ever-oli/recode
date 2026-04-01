import torch


class BatchNorm:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.gamma = torch.ones(num_features, device=device)
        self.beta = torch.zeros(num_features, device=device)
        self.running_mean = torch.zeros(num_features, device=device)
        self.running_var = torch.ones(num_features, device=device)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        x = x.to(self.device)
        original_shape = x.shape

        if len(original_shape) > 2:
            batch, channels = original_shape[0], original_shape[1]
            x_reshaped = x.reshape(batch, channels, -1)
            x_reshaped = x_reshaped.permute(0, 2, 1).reshape(-1, channels)
        else:
            x_reshaped = x
            channels = original_shape[-1]

        if training:
            batch_mean = torch.mean(x_reshaped, dim=0)
            batch_var = torch.var(x_reshaped, dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_norm = (x_reshaped - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x_reshaped - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        out = self.gamma * x_norm + self.beta

        if len(original_shape) > 2:
            out = out.reshape(batch, -1, channels).permute(0, 2, 1)
            out = out.reshape(original_shape)
        else:
            out = out.reshape(original_shape)

        return out


def relu(x: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    return torch.maximum(torch.tensor(0.0, device=device), x)


def post_activation_block(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, bn1: BatchNorm, bn2: BatchNorm) -> torch.Tensor:
    out = torch.matmul(x, W1)
    out = bn1.forward(out)
    out = relu(out, device=bn1.device)
    out = torch.matmul(out, W2)
    out = bn2.forward(out)
    return relu(out + x, device=bn1.device)


def pre_activation_block(x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, bn1: BatchNorm, bn2: BatchNorm) -> torch.Tensor:
    out = bn1.forward(x)
    out = relu(out, device=bn1.device)
    out = torch.matmul(out, W1)
    out = bn2.forward(out)
    out = relu(out, device=bn1.device)
    out = torch.matmul(out, W2)
    return out + x
