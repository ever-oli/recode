import torch


def ddpm_sample(model_predict: callable, shape: tuple, betas: torch.Tensor, T: int, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    betas = betas.to(device)
    x_t = torch.randn(*shape, device=device)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    for t in range(T, 0, -1):
        epsilon_pred = model_predict(x_t, t)

        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]

        inv_sqrt_alpha_t = 1.0 / torch.sqrt(alpha_t)
        noise_coeff = beta_t / torch.sqrt(1.0 - alpha_bar_t)

        mu = inv_sqrt_alpha_t * (x_t - noise_coeff * epsilon_pred)

        if t > 1:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn(*shape, device=device)
            x_t = mu + sigma_t * z
        else:
            x_t = mu

    return x_t
