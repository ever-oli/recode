import torch


def rnn_forward(X: torch.Tensor, h_0: torch.Tensor, W_xh: torch.Tensor, W_hh: torch.Tensor, b_h: torch.Tensor) -> tuple:
    batch_size, time_steps, _ = X.shape
    h_current = h_0
    h_all_list = []

    for t in range(time_steps):
        x_t = X[:, t, :]
        h_current = torch.tanh(torch.matmul(x_t, W_xh.T) + torch.matmul(h_current, W_hh.T) + b_h)
        h_all_list.append(h_current)

    h_all = torch.stack(h_all_list, dim=1)
    h_final = h_current
    return h_all, h_final
