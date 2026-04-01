import mlx.core as mx


class BatchNorm:
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = mx.ones((num_features,))
        self.beta = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def forward(self, x: mx.array, training: bool = True) -> mx.array:
        original_shape = x.shape

        if len(original_shape) > 2:
            batch, channels = original_shape[0], original_shape[1]
            x_reshaped = mx.reshape(x, (batch, channels, -1))
            x_reshaped = mx.reshape(mx.transpose(x_reshaped, (0, 2, 1)), (-1, channels))
        else:
            x_reshaped = x
            channels = original_shape[-1]

        if training:
            batch_mean = mx.mean(x_reshaped, axis=0)
            batch_var = mx.var(x_reshaped, axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            x_norm = (x_reshaped - batch_mean) / mx.sqrt(batch_var + self.eps)
        else:
            x_norm = (x_reshaped - self.running_mean) / mx.sqrt(self.running_var + self.eps)

        out = self.gamma * x_norm + self.beta

        if len(original_shape) > 2:
            out = mx.reshape(out, (batch, -1, channels))
            out = mx.transpose(out, (0, 2, 1))
            out = mx.reshape(out, original_shape)
        else:
            out = mx.reshape(out, original_shape)

        return out


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)


def post_activation_block(x: mx.array, W1: mx.array, W2: mx.array, bn1: BatchNorm, bn2: BatchNorm) -> mx.array:
    out = mx.matmul(x, W1)
    out = bn1.forward(out)
    out = relu(out)
    out = mx.matmul(out, W2)
    out = bn2.forward(out)
    return relu(out + x)


def pre_activation_block(x: mx.array, W1: mx.array, W2: mx.array, bn1: BatchNorm, bn2: BatchNorm) -> mx.array:
    out = bn1.forward(x)
    out = relu(out)
    out = mx.matmul(out, W1)
    out = bn2.forward(out)
    out = relu(out)
    out = mx.matmul(out, W2)
    return out + x
