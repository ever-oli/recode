import torch


class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.hidden_dim = hidden_dim
        self.W_xh = torch.randn(hidden_dim, input_dim, device=device) * torch.sqrt(torch.tensor(2.0 / (input_dim + hidden_dim), device=device))
        self.W_hh = torch.randn(hidden_dim, hidden_dim, device=device) * torch.sqrt(torch.tensor(2.0 / (2 * hidden_dim), device=device))
        self.W_hy = torch.randn(output_dim, hidden_dim, device=device) * torch.sqrt(torch.tensor(2.0 / (hidden_dim + output_dim), device=device))
        self.b_h = torch.zeros(hidden_dim, device=device)
        self.b_y = torch.zeros(output_dim, device=device)

    def forward(self, X: torch.Tensor, h_0: torch.Tensor = None) -> tuple:
        X = X.to(self.device)
        batch_size, time_steps, _ = X.shape
        if h_0 is None:
            h_current = torch.zeros((batch_size, self.hidden_dim), device=self.device)
        else:
            h_current = h_0.to(self.device)

        h_list = []
        for t in range(time_steps):
            x_t = X[:, t, :]
            h_current = torch.tanh(torch.matmul(x_t, self.W_xh.T) + torch.matmul(h_current, self.W_hh.T) + self.b_h)
            h_list.append(h_current)

        h_seq = torch.stack(h_list, dim=1)
        h_final = h_current

        h_flat = h_seq.reshape(-1, self.hidden_dim)
        y_flat = torch.matmul(h_flat, self.W_hy.T) + self.b_y
        y_seq = y_flat.reshape(batch_size, time_steps, -1)

        return y_seq, h_final
