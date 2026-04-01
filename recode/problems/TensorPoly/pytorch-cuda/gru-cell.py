import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


def gru_cell(x_t: torch.Tensor, h_prev: torch.Tensor,
             W_r: torch.Tensor, W_z: torch.Tensor, W_h: torch.Tensor,
             b_r: torch.Tensor, b_z: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    concat_gates = torch.cat([h_prev, x_t], dim=-1)
    r_t = sigmoid(torch.matmul(concat_gates, W_r.T) + b_r)
    z_t = sigmoid(torch.matmul(concat_gates, W_z.T) + b_z)

    gated_h = r_t * h_prev
    concat_cand = torch.cat([gated_h, x_t], dim=-1)
    h_tilde = torch.tanh(torch.matmul(concat_cand, W_h.T) + b_h)

    h_t = z_t * h_prev + (1 - z_t) * h_tilde
    return h_t
