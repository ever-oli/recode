import torch


def encoder_block(x: torch.Tensor, out_channels: int) -> tuple:
    batch, H, W, _ = x.shape
    skip_H = H - 4
    skip_W = W - 4
    skip = torch.zeros((batch, skip_H, skip_W, out_channels))
    pool_H = skip_H // 2
    pool_W = skip_W // 2
    pooled = torch.zeros((batch, pool_H, pool_W, out_channels))
    return pooled, skip


def bottleneck(x: torch.Tensor, out_channels: int) -> torch.Tensor:
    batch, H, W, _ = x.shape
    return torch.zeros((batch, H - 4, W - 4, out_channels))


def decoder_block(x: torch.Tensor, skip: torch.Tensor, out_channels: int) -> torch.Tensor:
    batch, H, W, _ = x.shape
    H_up = H * 2
    W_up = W * 2

    _, H_skip, W_skip, _ = skip.shape
    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    _ = skip[:, crop_h:crop_h + H_up, crop_w:crop_w + W_up, :]

    H_out = H_up - 4
    W_out = W_up - 4
    return torch.zeros((batch, H_out, W_out, out_channels))


def output_layer(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    batch, H, W, _ = x.shape
    return torch.zeros((batch, H, W, num_classes))


def unet(x: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    e1_pool, e1_skip = encoder_block(x, out_channels=64)
    e2_pool, e2_skip = encoder_block(e1_pool, out_channels=128)
    e3_pool, e3_skip = encoder_block(e2_pool, out_channels=256)
    e4_pool, e4_skip = encoder_block(e3_pool, out_channels=512)

    bottleneck_out = bottleneck(e4_pool, out_channels=1024)

    d4_out = decoder_block(bottleneck_out, e4_skip, out_channels=512)
    d3_out = decoder_block(d4_out, e3_skip, out_channels=256)
    d2_out = decoder_block(d3_out, e2_skip, out_channels=128)
    d1_out = decoder_block(d2_out, e1_skip, out_channels=64)

    return output_layer(d1_out, num_classes)
