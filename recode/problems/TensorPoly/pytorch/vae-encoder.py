import torch


def vae_encoder(x: torch.Tensor, latent_dim: int) -> tuple:
    _, input_dim = x.shape
    hidden_dim = 256

    w_h = torch.randn(input_dim, hidden_dim) * 0.01
    b_h = torch.zeros(hidden_dim)
    h = torch.maximum(torch.tensor(0.0), torch.matmul(x, w_h) + b_h)

    w_mu = torch.randn(hidden_dim, latent_dim) * 0.01
    b_mu = torch.zeros(latent_dim)
    mu = torch.matmul(h, w_mu) + b_mu

    w_log_var = torch.randn(hidden_dim, latent_dim) * 0.01
    b_log_var = torch.zeros(latent_dim)
    log_var = torch.matmul(h, w_log_var) + b_log_var

    return mu, log_var
