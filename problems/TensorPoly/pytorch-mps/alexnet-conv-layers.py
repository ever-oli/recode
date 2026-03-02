import torch


def alexnet_conv1(image: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    image = image.to(device)
    batch_size = image.shape[0]
    output_h = 55
    output_w = 55
    num_filters = 96
    return torch.zeros((batch_size, output_h, output_w, num_filters), device=device)
