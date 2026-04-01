import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -500, 500)
    return 1 / (1 + torch.exp(-x))


class GAN:
    def __init__(self, data_dim: int, noise_dim: int, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        self.G_W1 = torch.randn(noise_dim, 128, device=self.device) * 0.02
        self.G_b1 = torch.zeros(128, device=self.device)
        self.G_W2 = torch.randn(128, data_dim, device=self.device) * 0.02
        self.G_b2 = torch.zeros(data_dim, device=self.device)

        self.D_W1 = torch.randn(data_dim, 256, device=self.device) * 0.02
        self.D_b1 = torch.zeros(256, device=self.device)
        self.D_W2 = torch.randn(256, 128, device=self.device) * 0.02
        self.D_b2 = torch.zeros(128, device=self.device)
        self.D_W3 = torch.randn(128, 1, device=self.device) * 0.02
        self.D_b3 = torch.zeros(1, device=self.device)

        self.d_lr = 0.001
        self.g_lr = 0.001

    def _generator_forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.maximum(torch.tensor(0.0, device=self.device), torch.matmul(z, self.G_W1) + self.G_b1)
        return torch.tanh(torch.matmul(h, self.G_W2) + self.G_b2)

    def _discriminator_forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.matmul(x, self.D_W1) + self.D_b1
        h1 = torch.maximum(0.2 * h1, h1)
        h2 = torch.matmul(h1, self.D_W2) + self.D_b2
        h2 = torch.maximum(0.2 * h2, h2)
        logits = torch.matmul(h2, self.D_W3) + self.D_b3
        return sigmoid(logits).flatten()

    def generate(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.noise_dim, device=self.device)
        return self._generator_forward(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self._discriminator_forward(x)

    def train_step(self, real_data: torch.Tensor) -> dict:
        real_data = real_data.to(self.device)
        batch_size = real_data.shape[0]
        eps = 1e-8

        fake_data = self.generate(batch_size)
        real_probs = self.discriminate(real_data)
        fake_probs = self.discriminate(fake_data)

        d_loss = -torch.mean(torch.log(real_probs + eps) + torch.log(1.0 - fake_probs + eps))
        g_loss = -torch.mean(torch.log(fake_probs + eps))

        return {
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
        }
