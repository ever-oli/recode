import numpy as np


def compute_ddpm_loss(model_predict: callable, x_0: np.ndarray, betas: np.ndarray, T: int) -> float:
    batch_size = x_0.shape[0]
    t = np.random.randint(1, T + 1, size=(batch_size,))

    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    a_bar_t = alpha_bars[t - 1]

    broadcast_shape = [-1] + [1] * (x_0.ndim - 1)
    a_bar_t = a_bar_t.reshape(broadcast_shape)

    epsilon = np.random.randn(*x_0.shape)
    x_t = np.sqrt(a_bar_t) * x_0 + np.sqrt(1.0 - a_bar_t) * epsilon

    epsilon_pred = model_predict(x_t, t)
    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return float(loss)
