import numpy as np


def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Sample from latent distribution using reparameterization trick.
    """
    std = np.exp(0.5 * log_var)
    epsilon = np.random.randn(*mu.shape)
    return mu + std * epsilon
