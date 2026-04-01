function vgg_conv_block(x, num_convs::Int, out_channels::Int)
    current_x = x
    for _ in 1:num_convs
        in_channels = size(current_x, 4)
        limit = sqrt(2 / (3 * 3 * in_channels))
        weights = randn(3, 3, in_channels, out_channels) .* limit
        bias = zeros(out_channels)

        batch, h, w, _ = size(current_x)
        padded_x = zeros(batch, h + 2, w + 2, in_channels)
        padded_x[:, 2:(h + 1), 2:(w + 1), :] .= current_x
        out = zeros(batch, h, w, out_channels)

        for i in 1:3
            for j in 1:3
                window = padded_x[:, i:(i + h - 1), j:(j + w - 1), :]
                for b in 1:batch
                    for r in 1:h
                        for c in 1:w
                            out[b, r, c, :] .+= window[b, r, c, :] * weights[i, j, :, :]
                        end
                    end
                end
            end
        end

        out .+= reshape(bias, 1, 1, 1, :)
        current_x = max.(0, out)
    end
    current_x
end
