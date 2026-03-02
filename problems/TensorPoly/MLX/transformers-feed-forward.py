import mlx.core as mx


def feed_forward(x: mx.array, W1: mx.array, b1: mx.array, W2: mx.array, b2: mx.array) -> mx.array:
    hidden = mx.matmul(x, W1) + b1
    relu_out = mx.maximum(0, hidden)
    return mx.matmul(relu_out, W2) + b2
