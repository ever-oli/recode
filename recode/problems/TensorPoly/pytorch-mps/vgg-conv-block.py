import torch


def vgg_conv_block(x: torch.Tensor, num_convs: int, out_channels: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    current_x = x.to(device)
    for _ in range(num_convs):
        _, _, _, c = current_x.shape
        limit = torch.sqrt(torch.tensor(2.0 / (3 * 3 * c), device=device))
        weights = torch.randn(3, 3, c, out_channels, device=device) * limit
        bias = torch.zeros(out_channels, device=device)

        batch, h, w, _ = current_x.shape
        padded_x = torch.zeros((batch, h + 2, w + 2, c), device=device)
        padded_x[:, 1:h + 1, 1:w + 1, :] = current_x

        out = torch.zeros((batch, h, w, out_channels), device=device)
        for i in range(3):
            for j in range(3):
                window = padded_x[:, i:i + h, j:j + w, :]
                out = out + torch.tensordot(window, weights[i, j], dims=([3], [0]))

        out = out + bias
        current_x = torch.maximum(torch.tensor(0.0, device=device), out)

    return current_x
