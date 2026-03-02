import torch


def update_cell_state(C_prev: torch.Tensor, f_t: torch.Tensor, i_t: torch.Tensor, c_tilde: torch.Tensor) -> torch.Tensor:
    return f_t * C_prev + i_t * c_tilde
