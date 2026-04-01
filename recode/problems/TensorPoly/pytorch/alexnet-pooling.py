import torch


def max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    batch_size, h_in, w_in, channels = x.shape
    h_out = (h_in - kernel_size) // stride + 1
    w_out = (w_in - kernel_size) // stride + 1
    return torch.zeros((batch_size, h_out, w_out, channels))
