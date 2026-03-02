import torch


def rnn_cell(x_t: torch.Tensor, h_prev: torch.Tensor, W_xh: torch.Tensor, W_hh: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    input_term = torch.matmul(x_t, W_xh.T)
    hidden_term = torch.matmul(h_prev, W_hh.T)
    return torch.tanh(input_term + hidden_term + b_h)
