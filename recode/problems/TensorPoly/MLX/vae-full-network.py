import mlx.core as mx


class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 256

        self.w_enc = mx.random.normal(shape=(input_dim, self.hidden_dim)) * 0.01
        self.b_enc = mx.zeros((self.hidden_dim,))

        self.w_mu = mx.random.normal(shape=(self.hidden_dim, latent_dim)) * 0.01
        self.b_mu = mx.zeros((latent_dim,))
        self.w_log_var = mx.random.normal(shape=(self.hidden_dim, latent_dim)) * 0.01
        self.b_log_var = mx.zeros((latent_dim,))

        self.w_dec_h = mx.random.normal(shape=(latent_dim, self.hidden_dim)) * 0.01
        self.b_dec_h = mx.zeros((self.hidden_dim,))
        self.w_dec_out = mx.random.normal(shape=(self.hidden_dim, input_dim)) * 0.01
        self.b_dec_out = mx.zeros((input_dim,))

    def forward(self, x: mx.array) -> tuple:
        h_enc = mx.maximum(0, mx.matmul(x, self.w_enc) + self.b_enc)
        mu = mx.matmul(h_enc, self.w_mu) + self.b_mu
        log_var = mx.matmul(h_enc, self.w_log_var) + self.b_log_var

        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(shape=mu.shape)
        z = mu + std * eps

        h_dec = mx.maximum(0, mx.matmul(z, self.w_dec_h) + self.b_dec_h)
        logits = mx.matmul(h_dec, self.w_dec_out) + self.b_dec_out
        x_recon = 1 / (1 + mx.exp(-logits))

        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> mx.array:
        z = mx.random.normal(shape=(n_samples, self.latent_dim))
        h_dec = mx.maximum(0, mx.matmul(z, self.w_dec_h) + self.b_dec_h)
        logits = mx.matmul(h_dec, self.w_dec_out) + self.b_dec_out
        return 1 / (1 + mx.exp(-logits))
