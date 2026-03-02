import torch


def layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device)) * (x + 0.044715 * x ** 3)))


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=axis)


def multi_head_self_attention(x: torch.Tensor, num_heads: int, embed_dim: int) -> torch.Tensor:
    batch, seq_len, _ = x.shape
    head_dim = embed_dim // num_heads

    W_q = torch.randn(embed_dim, embed_dim, device=x.device) * 0.02
    W_k = torch.randn(embed_dim, embed_dim, device=x.device) * 0.02
    W_v = torch.randn(embed_dim, embed_dim, device=x.device) * 0.02
    W_o = torch.randn(embed_dim, embed_dim, device=x.device) * 0.02

    Q = torch.matmul(x, W_q)
    K = torch.matmul(x, W_k)
    V = torch.matmul(x, W_v)

    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=x.dtype, device=x.device))
    attn_weights = softmax(scores, axis=-1)
    attn_output = torch.matmul(attn_weights, V)

    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, embed_dim)
    return torch.matmul(attn_output, W_o)


def mlp(x: torch.Tensor, embed_dim: int, mlp_ratio: float) -> torch.Tensor:
    hidden_dim = int(embed_dim * mlp_ratio)
    W1 = torch.randn(embed_dim, hidden_dim, device=x.device) * 0.02
    b1 = torch.zeros(hidden_dim, device=x.device)
    W2 = torch.randn(hidden_dim, embed_dim, device=x.device) * 0.02
    b2 = torch.zeros(embed_dim, device=x.device)

    h = gelu(torch.matmul(x, W1) + b1)
    return torch.matmul(h, W2) + b2


def vit_encoder_block(x: torch.Tensor, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> torch.Tensor:
    x_norm1 = layer_norm(x)
    attn_output = multi_head_self_attention(x_norm1, num_heads, embed_dim)
    x = x + attn_output

    x_norm2 = layer_norm(x)
    mlp_output = mlp(x_norm2, embed_dim, mlp_ratio)
    x = x + mlp_output
    return x
