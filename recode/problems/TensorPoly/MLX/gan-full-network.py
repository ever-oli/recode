import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    x = mx.clip(x, -500, 500)
    return 1 / (1 + mx.exp(-x))


class GAN:
    def __init__(self, data_dim: int, noise_dim: int):
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        self.G_W1 = mx.random.normal(shape=(noise_dim, 128)) * 0.02
        self.G_b1 = mx.zeros((128,))
        self.G_W2 = mx.random.normal(shape=(128, data_dim)) * 0.02
        self.G_b2 = mx.zeros((data_dim,))

        self.D_W1 = mx.random.normal(shape=(data_dim, 256)) * 0.02
        self.D_b1 = mx.zeros((256,))
        self.D_W2 = mx.random.normal(shape=(256, 128)) * 0.02
        self.D_b2 = mx.zeros((128,))
        self.D_W3 = mx.random.normal(shape=(128, 1)) * 0.02
        self.D_b3 = mx.zeros((1,))

        self.d_lr = 0.001
        self.g_lr = 0.001

    def _generator_forward(self, z: mx.array) -> mx.array:
        h = mx.maximum(0, mx.matmul(z, self.G_W1) + self.G_b1)
        return mx.tanh(mx.matmul(h, self.G_W2) + self.G_b2)

    def _discriminator_forward(self, x: mx.array) -> mx.array:
        h1 = mx.matmul(x, self.D_W1) + self.D_b1
        h1 = mx.maximum(0.2 * h1, h1)
        h2 = mx.matmul(h1, self.D_W2) + self.D_b2
        h2 = mx.maximum(0.2 * h2, h2)
        logits = mx.matmul(h2, self.D_W3) + self.D_b3
        return mx.reshape(sigmoid(logits), (-1,))

    def generate(self, n: int) -> mx.array:
        z = mx.random.normal(shape=(n, self.noise_dim))
        return self._generator_forward(z)

    def discriminate(self, x: mx.array) -> mx.array:
        return self._discriminator_forward(x)

    def train_step(self, real_data: mx.array) -> dict:
        batch_size = real_data.shape[0]
        eps = 1e-8

        fake_data = self.generate(batch_size)
        real_probs = self.discriminate(real_data)
        fake_probs = self.discriminate(fake_data)

        d_loss = -mx.mean(mx.log(real_probs + eps) + mx.log(1.0 - fake_probs + eps))
        g_loss = -mx.mean(mx.log(fake_probs + eps))

        return {
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
        }
