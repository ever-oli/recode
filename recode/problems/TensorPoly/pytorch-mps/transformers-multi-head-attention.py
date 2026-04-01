import torch


def softmax(x, axis=-1):
    return torch.softmax(x, dim=axis)


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
    scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    attention_weights = softmax(scaled_scores, axis=-1)
    head_outputs = torch.matmul(attention_weights, V_trans)

    head_outputs_trans = head_outputs.transpose(1, 2)
    concatenated = head_outputs_trans.reshape(batch_size, seq_len, d_model)
    return torch.matmul(concatenated, W_o)
