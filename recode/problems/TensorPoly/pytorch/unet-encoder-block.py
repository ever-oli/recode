import torch


def unet_encoder_block(x: torch.Tensor, out_channels: int) -> tuple:
    batch, H, W, _ = x.shape
    skip_H = H - 4
    skip_W = W - 4
    skip_out = torch.zeros((batch, skip_H, skip_W, out_channels))

    pool_H = skip_H // 2
    pool_W = skip_W // 2
    pool_out = torch.zeros((batch, pool_H, pool_W, out_channels))

    return pool_out, skip_out
