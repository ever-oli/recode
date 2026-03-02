import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        batch_size, seq_len, _ = X.shape
        h_t = np.zeros((batch_size, self.hidden_dim))
        c_t = np.zeros((batch_size, self.hidden_dim))

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = np.concatenate([h_t, x_t], axis=1)

            f_t = sigmoid(concat @ self.W_f.T + self.b_f)
            i_t = sigmoid(concat @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(concat @ self.W_c.T + self.b_c)
            o_t = sigmoid(concat @ self.W_o.T + self.b_o)

            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * np.tanh(c_t)

            h_states.append(h_t)

        h_all = np.stack(h_states, axis=1)
        h_flat = h_all.reshape(-1, self.hidden_dim)
        y_flat = h_flat @ self.W_y.T + self.b_y
        y = y_flat.reshape(batch_size, seq_len, -1)

        return y, h_t, c_t
