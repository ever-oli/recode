function encoder_block(x, out_channels::Int)
    batch, H, W, _ = size(x)
    skip_H = H - 4
    skip_W = W - 4
    skip = zeros(batch, skip_H, skip_W, out_channels)
    pool_H = skip_H ÷ 2
    pool_W = skip_W ÷ 2
    pooled = zeros(batch, pool_H, pool_W, out_channels)
    return pooled, skip
end


function bottleneck(x, out_channels::Int)
    batch, H, W, _ = size(x)
    return zeros(batch, H - 4, W - 4, out_channels)
end


function decoder_block(x, skip, out_channels::Int)
    batch, H, W, _ = size(x)
    H_up = H * 2
    W_up = W * 2

    _, H_skip, W_skip, _ = size(skip)
    crop_h = (H_skip - H_up) ÷ 2
    crop_w = (W_skip - W_up) ÷ 2
    _ = skip[:, (crop_h + 1):(crop_h + H_up), (crop_w + 1):(crop_w + W_up), :]

    H_out = H_up - 4
    W_out = W_up - 4
    return zeros(batch, H_out, W_out, out_channels)
end


function output_layer(x, num_classes::Int)
    batch, H, W, _ = size(x)
    return zeros(batch, H, W, num_classes)
end


function unet(x, num_classes::Int=2)
    e1_pool, e1_skip = encoder_block(x, 64)
    e2_pool, e2_skip = encoder_block(e1_pool, 128)
    e3_pool, e3_skip = encoder_block(e2_pool, 256)
    e4_pool, e4_skip = encoder_block(e3_pool, 512)

    bottleneck_out = bottleneck(e4_pool, 1024)

    d4_out = decoder_block(bottleneck_out, e4_skip, 512)
    d3_out = decoder_block(d4_out, e3_skip, 256)
    d2_out = decoder_block(d3_out, e2_skip, 128)
    d1_out = decoder_block(d2_out, e1_skip, 64)

    return output_layer(d1_out, num_classes)
end
