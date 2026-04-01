import numpy as np


def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    batch, H, W, _ = x.shape
    _, H_skip, W_skip, _ = skip.shape

    H_up = H * 2
    W_up = W * 2

    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    _ = skip[:, crop_h:crop_h + H_up, crop_w:crop_w + W_up, :]

    H_out = H_up - 4
    W_out = W_up - 4

    output = np.zeros((batch, H_out, W_out, out_channels))
    return output
