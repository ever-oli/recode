function unet_decoder_block(x, skip, out_channels::Int)
    batch, H, W, _ = size(x)
    _, H_skip, W_skip, _ = size(skip)

    H_up = H * 2
    W_up = W * 2

    crop_h = (H_skip - H_up) ÷ 2
    crop_w = (W_skip - W_up) ÷ 2
    _ = skip[:, (crop_h + 1):(crop_h + H_up), (crop_w + 1):(crop_w + W_up), :]

    H_out = H_up - 4
    W_out = W_up - 4
    return zeros(batch, H_out, W_out, out_channels)
end
