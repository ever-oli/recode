function conv_relu(x, out_channels)
    _, _, _, C = size(x)
    W_weights = randn(C, out_channels) .* 0.1
    x_proj = reshape(x, :, C) * W_weights
    x_proj = reshape(x_proj, size(x, 1), size(x, 2), size(x, 3), out_channels)
    max.(0, x_proj)
end

function maxpool_2x2(x)
    B, H, W, C = size(x)
    reshaped = reshape(x, B, div(H, 2), 2, div(W, 2), 2, C)
    maximum(reshaped, dims=(3, 5))
end

function vgg_features(x, config)
    out = x
    for layer in config
        if layer isa Int
            out = conv_relu(out, layer)
        elseif layer == "M"
            out = maxpool_2x2(out)
        end
    end
    out
end
