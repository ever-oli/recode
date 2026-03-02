import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


def update_gate(h_prev: torch.Tensor, x_t: torch.Tensor, W_z: torch.Tensor, b_z: torch.Tensor) -> torch.Tensor:
    concat = torch.cat([h_prev, x_t], dim=-1)
    linear_transform = torch.matmul(concat, W_z.T) + b_z
    return sigmoid(linear_transform)
