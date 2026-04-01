import torch


def max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    batch_size, h_in, w_in, channels = x.shape
    h_out = (h_in - kernel_size) // stride + 1
    w_out = (w_in - kernel_size) // stride + 1
    return torch.zeros((batch_size, h_out, w_out, channels), device=device)
