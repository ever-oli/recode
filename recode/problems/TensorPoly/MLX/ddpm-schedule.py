import mlx.core as mx


def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> mx.array:
    return mx.linspace(beta_1, beta_T, T)


def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> mx.array:
    t = mx.arange(1, T + 1)
    f_0 = mx.cos(s / (1 + s) * mx.pi / 2) ** 2
    f_t = mx.cos(((t / T) + s) / (1 + s) * mx.pi / 2) ** 2
    return f_t / f_0


def alpha_bar_to_betas(alpha_bars: mx.array) -> mx.array:
    alpha_bars_prev = mx.concatenate([mx.array([1.0]), alpha_bars[:-1]])
    betas = 1.0 - (alpha_bars / alpha_bars_prev)
    return mx.clip(betas, 0.0, 0.999)
