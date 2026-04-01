import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = torch.sqrt(torch.tensor(2.0 / (input_dim + hidden_dim)))

        self.W_r = torch.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = torch.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = torch.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = torch.zeros(hidden_dim)
        self.b_z = torch.zeros(hidden_dim)
        self.b_h = torch.zeros(hidden_dim)

        self.W_y = torch.randn(output_dim, hidden_dim) * torch.sqrt(torch.tensor(2.0 / (hidden_dim + output_dim)))
        self.b_y = torch.zeros(output_dim)

    def forward(self, X: torch.Tensor) -> tuple:
        batch_size, seq_len, _ = X.shape
        h_t = torch.zeros((batch_size, self.hidden_dim))

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = torch.cat([h_t, x_t], dim=1)
            r_t = sigmoid(torch.matmul(concat, self.W_r.T) + self.b_r)
            z_t = sigmoid(torch.matmul(concat, self.W_z.T) + self.b_z)

            gated_h = r_t * h_t
            concat_cand = torch.cat([gated_h, x_t], dim=1)
            h_tilde = torch.tanh(torch.matmul(concat_cand, self.W_h.T) + self.b_h)

            h_t = z_t * h_t + (1 - z_t) * h_tilde
            h_states.append(h_t)

        h_all = torch.stack(h_states, dim=1)
        h_flat = h_all.reshape(-1, self.hidden_dim)
        y_flat = torch.matmul(h_flat, self.W_y.T) + self.b_y
        y = y_flat.reshape(batch_size, seq_len, -1)
        return y, h_t
