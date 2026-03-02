import torch


def bptt_single_step(dh_next: torch.Tensor, h_t: torch.Tensor, h_prev: torch.Tensor, x_t: torch.Tensor, W_hh: torch.Tensor) -> tuple:
    dtanh = (1 - h_t ** 2) * dh_next
    dW_hh = torch.matmul(dtanh.T, h_prev)
    dh_prev = torch.matmul(dtanh, W_hh)
    return dh_prev, dW_hh
