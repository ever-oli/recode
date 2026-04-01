import mlx.core as mx


def vae_encoder(x: mx.array, latent_dim: int) -> tuple:
    _, input_dim = x.shape
    hidden_dim = 256

    w_h = mx.random.normal(shape=(input_dim, hidden_dim)) * 0.01
    b_h = mx.zeros((hidden_dim,))
    h = mx.maximum(0, mx.matmul(x, w_h) + b_h)

    w_mu = mx.random.normal(shape=(hidden_dim, latent_dim)) * 0.01
    b_mu = mx.zeros((latent_dim,))
    mu = mx.matmul(h, w_mu) + b_mu

    w_log_var = mx.random.normal(shape=(hidden_dim, latent_dim)) * 0.01
    b_log_var = mx.zeros((latent_dim,))
    log_var = mx.matmul(h, w_log_var) + b_log_var

    return mu, log_var
