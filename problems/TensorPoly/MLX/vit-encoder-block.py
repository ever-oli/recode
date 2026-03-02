import mlx.core as mx


def layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1 + mx.tanh(mx.sqrt(mx.array(2 / mx.pi)) * (x + 0.044715 * x ** 3)))


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    x = x - mx.max(x, axis=axis, keepdims=True)
    e_x = mx.exp(x)
    return e_x / mx.sum(e_x, axis=axis, keepdims=True)


def multi_head_self_attention(x: mx.array, num_heads: int, embed_dim: int) -> mx.array:
    batch, seq_len, _ = x.shape
    head_dim = embed_dim // num_heads

    W_q = mx.random.normal(shape=(embed_dim, embed_dim)) * 0.02
    W_k = mx.random.normal(shape=(embed_dim, embed_dim)) * 0.02
    W_v = mx.random.normal(shape=(embed_dim, embed_dim)) * 0.02
    W_o = mx.random.normal(shape=(embed_dim, embed_dim)) * 0.02

    Q = mx.matmul(x, W_q)
    K = mx.matmul(x, W_k)
    V = mx.matmul(x, W_v)

    Q = mx.reshape(Q, (batch, seq_len, num_heads, head_dim))
    K = mx.reshape(K, (batch, seq_len, num_heads, head_dim))
    V = mx.reshape(V, (batch, seq_len, num_heads, head_dim))

    Q = mx.transpose(Q, (0, 2, 1, 3))
    K = mx.transpose(K, (0, 2, 1, 3))
    V = mx.transpose(V, (0, 2, 1, 3))

    scores = mx.matmul(Q, mx.transpose(K, (0, 1, 3, 2))) / mx.sqrt(mx.array(head_dim, dtype=x.dtype))
    attn_weights = softmax(scores, axis=-1)
    attn_output = mx.matmul(attn_weights, V)

    attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
    attn_output = mx.reshape(attn_output, (batch, seq_len, embed_dim))
    return mx.matmul(attn_output, W_o)


def mlp(x: mx.array, embed_dim: int, mlp_ratio: float) -> mx.array:
    hidden_dim = int(embed_dim * mlp_ratio)
    W1 = mx.random.normal(shape=(embed_dim, hidden_dim)) * 0.02
    b1 = mx.zeros((hidden_dim,))
    W2 = mx.random.normal(shape=(hidden_dim, embed_dim)) * 0.02
    b2 = mx.zeros((embed_dim,))

    h = gelu(mx.matmul(x, W1) + b1)
    return mx.matmul(h, W2) + b2


def vit_encoder_block(x: mx.array, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> mx.array:
    x_norm1 = layer_norm(x)
    attn_output = multi_head_self_attention(x_norm1, num_heads, embed_dim)
    x = x + attn_output

    x_norm2 = layer_norm(x)
    mlp_output = mlp(x_norm2, embed_dim, mlp_ratio)
    x = x + mlp_output

    return x
