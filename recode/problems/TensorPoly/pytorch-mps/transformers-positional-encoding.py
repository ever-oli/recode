import torch


def positional_encoding(seq_length: int, d_model: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    position = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    div_term = torch.exp(i * (-torch.log(torch.tensor(10000.0, device=device)) / d_model))

    pe = torch.zeros(seq_length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
