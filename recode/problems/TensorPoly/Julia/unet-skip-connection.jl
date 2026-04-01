function crop_and_concat(encoder_features, decoder_features)
    _, H_enc, W_enc, _ = size(encoder_features)
    _, H_dec, W_dec, _ = size(decoder_features)

    crop_h = (H_enc - H_dec) ÷ 2
    crop_w = (W_enc - W_dec) ÷ 2

    encoder_cropped = encoder_features[:, (crop_h + 1):(crop_h + H_dec), (crop_w + 1):(crop_w + W_dec), :]
    return cat(encoder_cropped, decoder_features; dims=4)
end
