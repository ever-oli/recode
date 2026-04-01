import torch


def generator(z: torch.Tensor, output_dim: int) -> torch.Tensor:
    _, noise_dim = z.shape

    W1 = torch.randn(noise_dim, 128) * 0.02
    b1 = torch.zeros(128)
    W2 = torch.randn(128, output_dim) * 0.02
    b2 = torch.zeros(output_dim)

    h1 = torch.maximum(torch.tensor(0.0), torch.matmul(z, W1) + b1)
    output = torch.tanh(torch.matmul(h1, W2) + b2)
    return output
