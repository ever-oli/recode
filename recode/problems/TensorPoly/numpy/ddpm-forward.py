import numpy as np


def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0)
    return alpha_bar


def forward_diffusion(x_0: np.ndarray, t: int, betas: np.ndarray) -> tuple:
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]

    epsilon = np.random.randn(*x_0.shape)

    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    return x_t, epsilon
