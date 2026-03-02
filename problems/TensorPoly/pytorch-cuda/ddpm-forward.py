import torch


def get_alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def forward_diffusion(x_0: torch.Tensor, t: int, betas: torch.Tensor, device=None) -> tuple:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x_0 = x_0.to(device)
    betas = betas.to(device)

    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]

    epsilon = torch.randn_like(x_0)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    return x_t, epsilon
