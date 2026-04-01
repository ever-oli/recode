import mlx.core as mx


def random_crop(image: mx.array, crop_size: int = 224) -> mx.array:
    h = image.shape[0]
    w = image.shape[1]

    max_top = h - crop_size
    max_left = w - crop_size
    top = int(mx.random.randint(0, max_top + 1).item())
    left = int(mx.random.randint(0, max_left + 1).item())
    return image[top:top + crop_size, left:left + crop_size, :]


def random_horizontal_flip(image: mx.array, p: float = 0.5) -> mx.array:
    if float(mx.random.uniform().item()) < p:
        return image[:, ::-1, :]
    return image
