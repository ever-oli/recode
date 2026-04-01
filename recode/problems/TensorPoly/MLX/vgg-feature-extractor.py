import mlx.core as mx


def conv_relu(x: mx.array, out_channels: int) -> mx.array:
    _, _, _, C = x.shape
    W_weights = mx.random.normal(shape=(C, out_channels)) * 0.1
    x = mx.matmul(x, W_weights)
    return mx.maximum(0, x)


def maxpool_2x2(x: mx.array) -> mx.array:
    B, H, W, C = x.shape
    reshaped_x = mx.reshape(x, (B, H // 2, 2, W // 2, 2, C))
    return mx.max(reshaped_x, axis=(2, 4))


def vgg_features(x: mx.array, config: list) -> mx.array:
    out = x
    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer)
        elif layer == "M":
            out = maxpool_2x2(out)
    return out
