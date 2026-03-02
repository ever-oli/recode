import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = mx.sqrt(mx.array(2.0 / (input_dim + hidden_dim)))

        self.W_r = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.W_z = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.W_h = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.b_r = mx.zeros((hidden_dim,))
        self.b_z = mx.zeros((hidden_dim,))
        self.b_h = mx.zeros((hidden_dim,))

        self.W_y = mx.random.normal(shape=(output_dim, hidden_dim)) * mx.sqrt(mx.array(2.0 / (hidden_dim + output_dim)))
        self.b_y = mx.zeros((output_dim,))

    def forward(self, X: mx.array) -> tuple:
        batch_size, seq_len, _ = X.shape
        h_t = mx.zeros((batch_size, self.hidden_dim))

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = mx.concatenate([h_t, x_t], axis=1)
            r_t = sigmoid(mx.matmul(concat, mx.transpose(self.W_r)) + self.b_r)
            z_t = sigmoid(mx.matmul(concat, mx.transpose(self.W_z)) + self.b_z)

            gated_h = r_t * h_t
            concat_cand = mx.concatenate([gated_h, x_t], axis=1)
            h_tilde = mx.tanh(mx.matmul(concat_cand, mx.transpose(self.W_h)) + self.b_h)

            h_t = z_t * h_t + (1 - z_t) * h_tilde
            h_states.append(h_t)

        h_all = mx.stack(h_states, axis=1)
        h_flat = mx.reshape(h_all, (-1, self.hidden_dim))
        y_flat = mx.matmul(h_flat, mx.transpose(self.W_y)) + self.b_y
        y = mx.reshape(y_flat, (batch_size, seq_len, -1))

        return y, h_t
