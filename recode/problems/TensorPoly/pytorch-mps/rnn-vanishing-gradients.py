import torch


def compute_gradient_norm_decay(T: int, W_hh: torch.Tensor) -> list:
    spectral_norm = torch.linalg.norm(W_hh, ord=2)
    norms = [1.0]
    current_norm = 1.0

    for _ in range(T - 1):
        current_norm *= float(spectral_norm)
        norms.append(current_norm)

    return norms
