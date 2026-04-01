import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = F.softmax(scaled_scores, dim=-1)
    return torch.matmul(attention_weights, V)
