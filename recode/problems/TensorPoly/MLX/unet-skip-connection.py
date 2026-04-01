import mlx.core as mx


def crop_and_concat(encoder_features: mx.array, decoder_features: mx.array) -> mx.array:
    _, H_enc, W_enc, _ = encoder_features.shape
    _, H_dec, W_dec, _ = decoder_features.shape

    crop_h = (H_enc - H_dec) // 2
    crop_w = (W_enc - W_dec) // 2

    encoder_cropped = encoder_features[:, crop_h:crop_h + H_dec, crop_w:crop_w + W_dec, :]
    return mx.concatenate([encoder_cropped, decoder_features], axis=-1)
