import mlx.core as mx


def layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def classification_head(encoder_output: mx.array, num_classes: int) -> mx.array:
    cls_token = encoder_output[:, 0, :]
    cls_norm = layer_norm(cls_token)

    embed_dim = cls_token.shape[-1]
    W = mx.random.normal(shape=(embed_dim, num_classes)) * 0.01
    b = mx.zeros((num_classes,))

    return mx.matmul(cls_norm, W) + b
