import numpy as np


class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 256

        self.w_enc = np.random.randn(input_dim, self.hidden_dim) * 0.01
        self.b_enc = np.zeros(self.hidden_dim)

        self.w_mu = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.w_log_var = np.random.randn(self.hidden_dim, latent_dim) * 0.01
        self.b_log_var = np.zeros(latent_dim)

        self.w_dec_h = np.random.randn(latent_dim, self.hidden_dim) * 0.01
        self.b_dec_h = np.zeros(self.hidden_dim)
        self.w_dec_out = np.random.randn(self.hidden_dim, input_dim) * 0.01
        self.b_dec_out = np.zeros(input_dim)

    def forward(self, x: np.ndarray) -> tuple:
        h_enc = np.maximum(0, x @ self.w_enc + self.b_enc)
        mu = h_enc @ self.w_mu + self.b_mu
        log_var = h_enc @ self.w_log_var + self.b_log_var

        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps

        h_dec = np.maximum(0, z @ self.w_dec_h + self.b_dec_h)
        logits = h_dec @ self.w_dec_out + self.b_dec_out
        x_recon = 1 / (1 + np.exp(-logits))

        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> np.ndarray:
        z = np.random.randn(n_samples, self.latent_dim)
        h_dec = np.maximum(0, z @ self.w_dec_h + self.b_dec_h)
        logits = h_dec @ self.w_dec_out + self.b_dec_out
        samples = 1 / (1 + np.exp(-logits))
        return samples
