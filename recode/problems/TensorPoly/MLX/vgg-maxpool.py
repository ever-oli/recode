import mlx.core as mx


def vgg_maxpool(x: mx.array) -> mx.array:
    batch, h, w, c = x.shape
    reshaped_x = mx.reshape(x, (batch, h // 2, 2, w // 2, 2, c))
    return mx.max(reshaped_x, axis=(2, 4))
