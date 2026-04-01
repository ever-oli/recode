import numpy as np


def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    var = np.exp(log_var)
    kl_element = 1 + log_var - np.square(mu) - var
    batch_kl = -0.5 * np.sum(kl_element, axis=1)
    return float(np.mean(batch_kl))
