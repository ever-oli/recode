import numpy as np


def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return image[top:top + crop_size, left:left + crop_size, :]


def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.random() < p:
        return image[:, ::-1, :]
    return image
