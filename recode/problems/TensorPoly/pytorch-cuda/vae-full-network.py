import torch


class VAE:
    def __init__(self, input_dim: int, latent_dim: int, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 256

        self.w_enc = torch.randn(input_dim, self.hidden_dim, device=self.device) * 0.01
        self.b_enc = torch.zeros(self.hidden_dim, device=self.device)

        self.w_mu = torch.randn(self.hidden_dim, latent_dim, device=self.device) * 0.01
        self.b_mu = torch.zeros(latent_dim, device=self.device)
        self.w_log_var = torch.randn(self.hidden_dim, latent_dim, device=self.device) * 0.01
        self.b_log_var = torch.zeros(latent_dim, device=self.device)

        self.w_dec_h = torch.randn(latent_dim, self.hidden_dim, device=self.device) * 0.01
        self.b_dec_h = torch.zeros(self.hidden_dim, device=self.device)
        self.w_dec_out = torch.randn(self.hidden_dim, input_dim, device=self.device) * 0.01
        self.b_dec_out = torch.zeros(input_dim, device=self.device)

    def forward(self, x: torch.Tensor) -> tuple:
        x = x.to(self.device)
        h_enc = torch.maximum(torch.tensor(0.0, device=self.device), torch.matmul(x, self.w_enc) + self.b_enc)
        mu = torch.matmul(h_enc, self.w_mu) + self.b_mu
        log_var = torch.matmul(h_enc, self.w_log_var) + self.b_log_var

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        z = mu + std * eps

        h_dec = torch.maximum(torch.tensor(0.0, device=self.device), torch.matmul(z, self.w_dec_h) + self.b_dec_h)
        logits = torch.matmul(h_dec, self.w_dec_out) + self.b_dec_out
        x_recon = 1 / (1 + torch.exp(-logits))

        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> torch.Tensor:
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        h_dec = torch.maximum(torch.tensor(0.0, device=self.device), torch.matmul(z, self.w_dec_h) + self.b_dec_h)
        logits = torch.matmul(h_dec, self.w_dec_out) + self.b_dec_out
        return 1 / (1 + torch.exp(-logits))
