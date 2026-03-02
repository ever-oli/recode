import torch


def softmax(x, axis=-1):
    return torch.softmax(x, dim=axis)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * x_normalized + beta


def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
                         W_o: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = torch.matmul(Q, W_q)
    K_proj = torch.matmul(K, W_k)
    V_proj = torch.matmul(V, W_v)

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    Q_trans = Q_heads.transpose(1, 2)
    K_trans = K_heads.transpose(1, 2)
    V_trans = V_heads.transpose(1, 2)

    scores = torch.matmul(Q_trans, K_trans.transpose(-2, -1))
    scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    attention_weights = softmax(scaled_scores, axis=-1)
    head_outputs = torch.matmul(attention_weights, V_trans)

    head_outputs_trans = head_outputs.transpose(1, 2)
    concatenated = head_outputs_trans.reshape(batch_size, seq_len, d_model)
    output = torch.matmul(concatenated, W_o)
    return output


def feed_forward(x: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor,
                 W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    hidden = torch.matmul(x, W1) + b1
    relu_out = torch.maximum(torch.tensor(0.0, dtype=hidden.dtype), hidden)
    return torch.matmul(relu_out, W2) + b2


def encoder_block(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
                  W_o: torch.Tensor, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor,
                  b2: torch.Tensor, gamma1: torch.Tensor, beta1: torch.Tensor,
                  gamma2: torch.Tensor, beta2: torch.Tensor, num_heads: int) -> torch.Tensor:
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_attn_residual = x + attn_output
    x_norm1 = layer_norm(x_attn_residual, gamma1, beta1)

    ff_output = feed_forward(x_norm1, W1, b1, W2, b2)
    x_ff_residual = x_norm1 + ff_output
    return layer_norm(x_ff_residual, gamma2, beta2)
