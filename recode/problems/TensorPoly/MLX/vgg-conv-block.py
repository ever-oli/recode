import mlx.core as mx


def vgg_conv_block(x: mx.array, num_convs: int, out_channels: int) -> mx.array:
    current_x = x
    for _ in range(num_convs):
        in_channels = current_x.shape[-1]
        limit = mx.sqrt(mx.array(2.0 / (3 * 3 * in_channels)))
        weights = mx.random.normal(shape=(3, 3, in_channels, out_channels)) * limit
        bias = mx.zeros((out_channels,))

        batch, h, w, _ = current_x.shape
        padded_x = mx.zeros((batch, h + 2, w + 2, in_channels))
        padded_x = padded_x.at[:, 1:h + 1, 1:w + 1, :].set(current_x)

        out = mx.zeros((batch, h, w, out_channels))
        for i in range(3):
            for j in range(3):
                window = padded_x[:, i:i + h, j:j + w, :]
                out = out + mx.tensordot(window, weights[i, j], axes=([3], [0]))

        out = out + bias
        current_x = mx.maximum(0, out)

    return current_x
