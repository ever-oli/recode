import torch


def sigmoid(x):
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    return 1.0 / (1.0 + torch.exp(-x_tensor))
