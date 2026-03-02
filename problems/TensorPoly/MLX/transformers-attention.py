import math
import mlx.core as mx


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    x = x - mx.max(x, axis=axis, keepdims=True)
    e_x = mx.exp(x)
    return e_x / mx.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: mx.array, K: mx.array, V: mx.array) -> mx.array:
    d_k = Q.shape[-1]
    scores = mx.matmul(Q, mx.swapaxes(K, -2, -1))
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis=-1)
    return mx.matmul(attention_weights, V)
