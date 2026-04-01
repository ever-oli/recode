import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-mx.clip(x, -500, 500)))


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = mx.sqrt(mx.array(2.0 / (input_dim + hidden_dim)))

        self.W_f = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.W_i = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.W_c = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.W_o = mx.random.normal(shape=(hidden_dim, hidden_dim + input_dim)) * scale
        self.b_f = mx.zeros((hidden_dim,))
        self.b_i = mx.zeros((hidden_dim,))
        self.b_c = mx.zeros((hidden_dim,))
        self.b_o = mx.zeros((hidden_dim,))

        self.W_y = mx.random.normal(shape=(output_dim, hidden_dim)) * mx.sqrt(mx.array(2.0 / (hidden_dim + output_dim)))
        self.b_y = mx.zeros((output_dim,))

    def forward(self, X: mx.array) -> tuple:
        batch_size, seq_len, _ = X.shape
        h_t = mx.zeros((batch_size, self.hidden_dim))
        c_t = mx.zeros((batch_size, self.hidden_dim))

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = mx.concatenate([h_t, x_t], axis=1)

            f_t = sigmoid(mx.matmul(concat, mx.transpose(self.W_f)) + self.b_f)
            i_t = sigmoid(mx.matmul(concat, mx.transpose(self.W_i)) + self.b_i)
            c_tilde = mx.tanh(mx.matmul(concat, mx.transpose(self.W_c)) + self.b_c)
            o_t = sigmoid(mx.matmul(concat, mx.transpose(self.W_o)) + self.b_o)

            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * mx.tanh(c_t)
            h_states.append(h_t)

        h_all = mx.stack(h_states, axis=1)
        h_flat = mx.reshape(h_all, (-1, self.hidden_dim))
        y_flat = mx.matmul(h_flat, mx.transpose(self.W_y)) + self.b_y
        y = mx.reshape(y_flat, (batch_size, seq_len, -1))

        return y, h_t, c_t
