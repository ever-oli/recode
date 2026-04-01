import numpy as np


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        batch_size, time_steps, _ = X.shape

        if h_0 is None:
            h_current = np.zeros((batch_size, self.hidden_dim))
        else:
            h_current = h_0

        h_list = []
        for t in range(time_steps):
            x_t = X[:, t, :]
            h_current = np.tanh(x_t @ self.W_xh.T + h_current @ self.W_hh.T + self.b_h)
            h_list.append(h_current)

        h_seq = np.stack(h_list, axis=1)
        h_final = h_current

        h_flat = h_seq.reshape(-1, self.hidden_dim)
        y_flat = h_flat @ self.W_hy.T + self.b_y
        y_seq = y_flat.reshape(batch_size, time_steps, -1)

        return y_seq, h_final
