import mlx.core as mx


def ddpm_sample(model_predict: callable, shape: tuple, betas: mx.array, T: int) -> mx.array:
    x_t = mx.random.normal(shape=shape)

    alphas = 1.0 - betas
    alpha_bars = mx.cumprod(alphas, axis=0)

    for t in range(T, 0, -1):
        epsilon_pred = model_predict(x_t, t)

        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]

        inv_sqrt_alpha_t = 1.0 / mx.sqrt(alpha_t)
        noise_coeff = beta_t / mx.sqrt(1.0 - alpha_bar_t)

        mu = inv_sqrt_alpha_t * (x_t - noise_coeff * epsilon_pred)

        if t > 1:
            sigma_t = mx.sqrt(beta_t)
            z = mx.random.normal(shape=shape)
            x_t = mu + sigma_t * z
        else:
            x_t = mu

    return x_t
