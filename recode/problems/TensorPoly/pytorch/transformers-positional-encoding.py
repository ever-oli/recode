import torch


def positional_encoding(seq_length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(i * (-torch.log(torch.tensor(10000.0)) / d_model))

    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
