import mlx.core as mx


def max_pool2d(x: mx.array, kernel_size: int = 3, stride: int = 2) -> mx.array:
    batch_size, h_in, w_in, channels = x.shape
    h_out = (h_in - kernel_size) // stride + 1
    w_out = (w_in - kernel_size) // stride + 1
    return mx.zeros((batch_size, h_out, w_out, channels))
