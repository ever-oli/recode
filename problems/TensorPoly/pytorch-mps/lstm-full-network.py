import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-torch.clamp(x, -500, 500)))


class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.hidden_dim = hidden_dim
        scale = torch.sqrt(torch.tensor(2.0 / (input_dim + hidden_dim), device=device))

        self.W_f = torch.randn(hidden_dim, hidden_dim + input_dim, device=device) * scale
        self.W_i = torch.randn(hidden_dim, hidden_dim + input_dim, device=device) * scale
        self.W_c = torch.randn(hidden_dim, hidden_dim + input_dim, device=device) * scale
        self.W_o = torch.randn(hidden_dim, hidden_dim + input_dim, device=device) * scale
        self.b_f = torch.zeros(hidden_dim, device=device)
        self.b_i = torch.zeros(hidden_dim, device=device)
        self.b_c = torch.zeros(hidden_dim, device=device)
        self.b_o = torch.zeros(hidden_dim, device=device)

        self.W_y = torch.randn(output_dim, hidden_dim, device=device) * torch.sqrt(torch.tensor(2.0 / (hidden_dim + output_dim), device=device))
        self.b_y = torch.zeros(output_dim, device=device)

    def forward(self, X: torch.Tensor) -> tuple:
        X = X.to(self.device)
        batch_size, seq_len, _ = X.shape
        h_t = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        c_t = torch.zeros((batch_size, self.hidden_dim), device=self.device)

        h_states = []
        for t in range(seq_len):
            x_t = X[:, t, :]
            concat = torch.cat([h_t, x_t], dim=1)

            f_t = sigmoid(torch.matmul(concat, self.W_f.T) + self.b_f)
            i_t = sigmoid(torch.matmul(concat, self.W_i.T) + self.b_i)
            c_tilde = torch.tanh(torch.matmul(concat, self.W_c.T) + self.b_c)
            o_t = sigmoid(torch.matmul(concat, self.W_o.T) + self.b_o)

            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)
            h_states.append(h_t)

        h_all = torch.stack(h_states, dim=1)
        h_flat = h_all.reshape(-1, self.hidden_dim)
        y_flat = torch.matmul(h_flat, self.W_y.T) + self.b_y
        y = y_flat.reshape(batch_size, seq_len, -1)

        return y, h_t, c_t
