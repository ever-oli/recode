import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


def output_gate(h_prev: torch.Tensor, x_t: torch.Tensor, C_t: torch.Tensor,
                W_o: torch.Tensor, b_o: torch.Tensor) -> tuple:
    concat = torch.cat([h_prev, x_t], dim=-1)
    o_t = sigmoid(torch.matmul(concat, W_o.T) + b_o)
    h_t = o_t * torch.tanh(C_t)
    return o_t, h_t
