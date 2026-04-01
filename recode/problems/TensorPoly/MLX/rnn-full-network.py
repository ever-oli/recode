import mlx.core as mx


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.W_xh = mx.random.normal(shape=(hidden_dim, input_dim)) * mx.sqrt(mx.array(2.0 / (input_dim + hidden_dim)))
        self.W_hh = mx.random.normal(shape=(hidden_dim, hidden_dim)) * mx.sqrt(mx.array(2.0 / (2 * hidden_dim)))
        self.W_hy = mx.random.normal(shape=(output_dim, hidden_dim)) * mx.sqrt(mx.array(2.0 / (hidden_dim + output_dim)))
        self.b_h = mx.zeros((hidden_dim,))
        self.b_y = mx.zeros((output_dim,))

    def forward(self, X: mx.array, h_0: mx.array = None) -> tuple:
        batch_size, time_steps, _ = X.shape
        if h_0 is None:
            h_current = mx.zeros((batch_size, self.hidden_dim))
        else:
            h_current = h_0

        h_list = []
        for t in range(time_steps):
            x_t = X[:, t, :]
            h_current = mx.tanh(mx.matmul(x_t, mx.transpose(self.W_xh)) + mx.matmul(h_current, mx.transpose(self.W_hh)) + self.b_h)
            h_list.append(h_current)

        h_seq = mx.stack(h_list, axis=1)
        h_final = h_current

        h_flat = mx.reshape(h_seq, (-1, self.hidden_dim))
        y_flat = mx.matmul(h_flat, mx.transpose(self.W_hy)) + self.b_y
        y_seq = mx.reshape(y_flat, (batch_size, time_steps, -1))

        return y_seq, h_final
