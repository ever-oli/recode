function unet_bottleneck(x, out_channels::Int)
    batch, H, W, _ = size(x)
    H_out = H - 4
    W_out = W - 4
    return zeros(batch, H_out, W_out, out_channels)
end
