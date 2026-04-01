import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        batch_size, seq_len, _ = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim))

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = np.concatenate([h_t, x_t], axis=1)
            r_t = sigmoid(concat @ self.W_r.T + self.b_r)
            z_t = sigmoid(concat @ self.W_z.T + self.b_z)

            gated_h = r_t * h_t
            concat_cand = np.concatenate([gated_h, x_t], axis=1)
            h_tilde = np.tanh(concat_cand @ self.W_h.T + self.b_h)

            h_t = z_t * h_t + (1 - z_t) * h_tilde
            h_states.append(h_t)

        h_all = np.stack(h_states, axis=1)
        h_flat = h_all.reshape(-1, self.hidden_dim)
        y_flat = h_flat @ self.W_y.T + self.b_y
        y = y_flat.reshape(batch_size, seq_len, -1)
        return y, h_t
