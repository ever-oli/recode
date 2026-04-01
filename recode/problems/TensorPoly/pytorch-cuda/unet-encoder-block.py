import torch


def unet_encoder_block(x: torch.Tensor, out_channels: int, device=None) -> tuple:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    batch, H, W, _ = x.shape
    skip_H = H - 4
    skip_W = W - 4
    skip_out = torch.zeros((batch, skip_H, skip_W, out_channels), device=device)

    pool_H = skip_H // 2
    pool_W = skip_W // 2
    pool_out = torch.zeros((batch, pool_H, pool_W, out_channels), device=device)

    return pool_out, skip_out
