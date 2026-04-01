import torch


def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.linspace(beta_1, beta_T, T, device=device)


def cosine_alpha_bar_schedule(T: int, s: float = 0.008, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    t = torch.arange(1, T + 1, device=device)
    f_0 = torch.cos(s / (1 + s) * torch.pi / 2) ** 2
    f_t = torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
    return f_t / f_0


def alpha_bar_to_betas(alpha_bars: torch.Tensor) -> torch.Tensor:
    alpha_bars_prev = torch.cat([torch.tensor([1.0], device=alpha_bars.device), alpha_bars[:-1]])
    betas = 1.0 - (alpha_bars / alpha_bars_prev)
    return torch.clamp(betas, 0.0, 0.999)
