import numpy as np


class BatchNorm:
    """Batch Normalization layer."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        original_shape = x.shape

        if len(original_shape) > 2:
            batch, channels = original_shape[0], original_shape[1]
            x_reshaped = x.reshape(batch, channels, -1)
            x_reshaped = x_reshaped.transpose(0, 2, 1).reshape(-1, channels)
        else:
            x_reshaped = x
            channels = original_shape[-1]

        if training:
            batch_mean = np.mean(x_reshaped, axis=0)
            batch_var = np.var(x_reshaped, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_norm = (x_reshaped - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * x_norm + self.beta

        if len(original_shape) > 2:
            out = out.reshape(batch, -1, channels).transpose(0, 2, 1)
            out = out.reshape(original_shape)
        else:
            out = out.reshape(original_shape)

        return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    out = np.matmul(x, W1)
    out = bn1.forward(out)
    out = relu(out)
    out = np.matmul(out, W2)
    out = bn2.forward(out)
    out = relu(out + x)
    return out


def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    out = bn1.forward(x)
    out = relu(out)
    out = np.matmul(out, W1)
    out = bn2.forward(out)
    out = relu(out)
    out = np.matmul(out, W2)
    return out + x
