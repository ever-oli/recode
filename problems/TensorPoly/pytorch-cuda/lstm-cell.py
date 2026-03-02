import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


def lstm_cell(x_t: torch.Tensor, h_prev: torch.Tensor, C_prev: torch.Tensor,
              W_f: torch.Tensor, W_i: torch.Tensor, W_c: torch.Tensor, W_o: torch.Tensor,
              b_f: torch.Tensor, b_i: torch.Tensor, b_c: torch.Tensor, b_o: torch.Tensor) -> tuple:
    concat = torch.cat([h_prev, x_t], dim=-1)
    f_t = sigmoid(torch.matmul(concat, W_f.T) + b_f)
    i_t = sigmoid(torch.matmul(concat, W_i.T) + b_i)
    c_tilde = torch.tanh(torch.matmul(concat, W_c.T) + b_c)
    o_t = sigmoid(torch.matmul(concat, W_o.T) + b_o)

    C_t = f_t * C_prev + i_t * c_tilde
    h_t = o_t * torch.tanh(C_t)
    return h_t, C_t
