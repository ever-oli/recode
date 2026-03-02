import numpy as np


def ddpm_sample(model_predict: callable, shape: tuple, betas: np.ndarray, T: int) -> np.ndarray:
    x_t = np.random.randn(*shape)

    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    for t in range(T, 0, -1):
        epsilon_pred = model_predict(x_t, t)

        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]

        inv_sqrt_alpha_t = 1.0 / np.sqrt(alpha_t)
        noise_coeff = beta_t / np.sqrt(1.0 - alpha_bar_t)

        mu = inv_sqrt_alpha_t * (x_t - noise_coeff * epsilon_pred)

        if t > 1:
            sigma_t = np.sqrt(beta_t)
            z = np.random.randn(*shape)
            x_t = mu + sigma_t * z
        else:
            x_t = mu

    return x_t
