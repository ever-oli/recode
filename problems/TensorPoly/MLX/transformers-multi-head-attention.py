import mlx.core as mx


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    x = x - mx.max(x, axis=axis, keepdims=True)
    e_x = mx.exp(x)
    return e_x / mx.sum(e_x, axis=axis, keepdims=True)


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
