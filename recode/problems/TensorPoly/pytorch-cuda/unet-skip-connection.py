import torch


def crop_and_concat(encoder_features: torch.Tensor, decoder_features: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder_features = encoder_features.to(device)
    decoder_features = decoder_features.to(device)
    _, H_enc, W_enc, _ = encoder_features.shape
    _, H_dec, W_dec, _ = decoder_features.shape

    crop_h = (H_enc - H_dec) // 2
    crop_w = (W_enc - W_dec) // 2

    encoder_cropped = encoder_features[:, crop_h:crop_h + H_dec, crop_w:crop_w + W_dec, :]
    return torch.cat([encoder_cropped, decoder_features], dim=-1)
