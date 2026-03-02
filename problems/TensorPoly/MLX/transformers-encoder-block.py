import mlx.core as mx


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    x = x - mx.max(x, axis=axis, keepdims=True)
    e_x = mx.exp(x)
    return e_x / mx.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: mx.array, gamma: mx.array, beta: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / mx.sqrt(variance + eps)
    return gamma * x_normalized + beta


def multi_head_attention(Q: mx.array, K: mx.array, V: mx.array,
                         W_q: mx.array, W_k: mx.array, W_v: mx.array,
                         W_o: mx.array, num_heads: int) -> mx.array:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = mx.matmul(Q, W_q)
    K_proj = mx.matmul(K, W_k)
    V_proj = mx.matmul(V, W_v)

    Q_heads = mx.reshape(Q_proj, (batch_size, seq_len, num_heads, d_k))
    K_heads = mx.reshape(K_proj, (batch_size, seq_len, num_heads, d_k))
    V_heads = mx.reshape(V_proj, (batch_size, seq_len, num_heads, d_k))

    Q_trans = mx.transpose(Q_heads, (0, 2, 1, 3))
    K_trans = mx.transpose(K_heads, (0, 2, 1, 3))
    V_trans = mx.transpose(V_heads, (0, 2, 1, 3))

    scores = mx.matmul(Q_trans, mx.transpose(K_trans, (0, 1, 3, 2)))
    scaled_scores = scores / mx.sqrt(mx.array(d_k, dtype=Q.dtype))
    attention_weights = softmax(scaled_scores, axis=-1)
    head_outputs = mx.matmul(attention_weights, V_trans)

    head_outputs_trans = mx.transpose(head_outputs, (0, 2, 1, 3))
    concatenated = mx.reshape(head_outputs_trans, (batch_size, seq_len, d_model))
    return mx.matmul(concatenated, W_o)


def feed_forward(x: mx.array, W1: mx.array, b1: mx.array, W2: mx.array, b2: mx.array) -> mx.array:
    hidden = mx.matmul(x, W1) + b1
    relu_out = mx.maximum(0, hidden)
    return mx.matmul(relu_out, W2) + b2


def encoder_block(x: mx.array, W_q: mx.array, W_k: mx.array, W_v: mx.array,
                  W_o: mx.array, W1: mx.array, b1: mx.array, W2: mx.array,
                  b2: mx.array, gamma1: mx.array, beta1: mx.array,
                  gamma2: mx.array, beta2: mx.array, num_heads: int) -> mx.array:
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_attn_residual = x + attn_output
    x_norm1 = layer_norm(x_attn_residual, gamma1, beta1)

    ff_output = feed_forward(x_norm1, W1, b1, W2, b2)
    x_ff_residual = x_norm1 + ff_output
    return layer_norm(x_ff_residual, gamma2, beta2)
