import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


def input_gate(h_prev: torch.Tensor, x_t: torch.Tensor,
               W_i: torch.Tensor, b_i: torch.Tensor,
               W_c: torch.Tensor, b_c: torch.Tensor) -> tuple:
    concat = torch.cat([h_prev, x_t], dim=-1)
    i_t = sigmoid(torch.matmul(concat, W_i.T) + b_i)
    c_tilde = torch.tanh(torch.matmul(concat, W_c.T) + b_c)
    return i_t, c_tilde
