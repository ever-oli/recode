import torch


def random_crop(image: torch.Tensor, crop_size: int = 224, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    image = image.to(device)
    h = image.shape[0]
    w = image.shape[1]
    top = torch.randint(0, h - crop_size + 1, (1,), device=device).item()
    left = torch.randint(0, w - crop_size + 1, (1,), device=device).item()
    return image[top:top + crop_size, left:left + crop_size, :]


def random_horizontal_flip(image: torch.Tensor, p: float = 0.5, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    image = image.to(device)
    if torch.rand(1, device=device).item() < p:
        return image[:, torch.arange(image.shape[1] - 1, -1, -1, device=device), :]
    return image
