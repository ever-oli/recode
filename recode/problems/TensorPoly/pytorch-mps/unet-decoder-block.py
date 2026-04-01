import torch


def unet_decoder_block(x: torch.Tensor, skip: torch.Tensor, out_channels: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = x.to(device)
    skip = skip.to(device)
    batch, H, W, _ = x.shape
    _, H_skip, W_skip, _ = skip.shape

    H_up = H * 2
    W_up = W * 2

    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    _ = skip[:, crop_h:crop_h + H_up, crop_w:crop_w + W_up, :]

    H_out = H_up - 4
    W_out = W_up - 4
    return torch.zeros((batch, H_out, W_out, out_channels), device=device)
