import numpy as np


def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    return np.linspace(beta_1, beta_T, T)


def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    t = np.arange(1, T + 1)
    f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
    f_t = np.cos(((t / T) + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bars = f_t / f_0
    return alpha_bars


def alpha_bar_to_betas(alpha_bars: np.ndarray) -> np.ndarray:
    alpha_bars_prev = np.concatenate(([1.0], alpha_bars[:-1]))
    betas = 1.0 - (alpha_bars / alpha_bars_prev)
    return np.clip(betas, 0.0, 0.999)
