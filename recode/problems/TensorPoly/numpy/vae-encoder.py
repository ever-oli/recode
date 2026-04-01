import numpy as np


def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    """
    Encode input to latent distribution parameters.
    """
    batch_size, input_dim = x.shape
    hidden_dim = 256

    w_h = np.random.randn(input_dim, hidden_dim) * 0.01
    b_h = np.zeros(hidden_dim)
    h = np.maximum(0, x @ w_h + b_h)

    w_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_mu = np.zeros(latent_dim)
    mu = h @ w_mu + b_mu

    w_log_var = np.random.randn(hidden_dim, latent_dim) * 0.01
    b_log_var = np.zeros(latent_dim)
    log_var = h @ w_log_var + b_log_var

    return mu, log_var
