function unet_encoder_block(x, out_channels::Int)
    batch, H, W, _ = size(x)
    skip_H = H - 4
    skip_W = W - 4
    skip_out = zeros(batch, skip_H, skip_W, out_channels)

    pool_H = skip_H ÷ 2
    pool_W = skip_W ÷ 2
    pool_out = zeros(batch, pool_H, pool_W, out_channels)

    return pool_out, skip_out
end
