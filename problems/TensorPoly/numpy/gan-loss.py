import numpy as np


def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    eps = 1e-8
    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)

    real_loss = -np.log(real_probs)
    fake_loss = -np.log(1 - fake_probs)
    total_loss = np.mean(real_loss + fake_loss)
    return float(total_loss)


def generator_loss(fake_probs: np.ndarray) -> float:
    eps = 1e-8
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    loss = -np.log(fake_probs)
    return float(np.mean(loss))
