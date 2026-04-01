import torch


def vgg_conv_block(x: torch.Tensor, num_convs: int, out_channels: int) -> torch.Tensor:
    current_x = x
    for _ in range(num_convs):
        _, _, _, c = current_x.shape
        limit = torch.sqrt(torch.tensor(2.0 / (3 * 3 * c)))
        weights = torch.randn(3, 3, c, out_channels) * limit
        bias = torch.zeros(out_channels)

        batch, h, w, _ = current_x.shape
        padded_x = torch.zeros((batch, h + 2, w + 2, c))
        padded_x[:, 1:h + 1, 1:w + 1, :] = current_x

        out = torch.zeros((batch, h, w, out_channels))
        for i in range(3):
            for j in range(3):
                window = padded_x[:, i:i + h, j:j + w, :]
                out = out + torch.tensordot(window, weights[i, j], dims=([3], [0]))

        out = out + bias
        current_x = torch.maximum(torch.tensor(0.0), out)

    return current_x
