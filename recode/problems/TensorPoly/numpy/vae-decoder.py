import numpy as np


def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    _, latent_dim = z.shape
    hidden_dim = 256

    w_h = np.random.randn(latent_dim, hidden_dim) * 0.01
    b_h = np.zeros(hidden_dim)
    h = np.maximum(0, z @ w_h + b_h)

    w_out = np.random.randn(hidden_dim, output_dim) * 0.01
    b_out = np.zeros(output_dim)
    logits = h @ w_out + b_out

    x_hat = 1 / (1 + np.exp(-logits))
    return x_hat
