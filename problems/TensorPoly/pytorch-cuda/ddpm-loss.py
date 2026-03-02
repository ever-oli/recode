import torch


def compute_ddpm_loss(model_predict: callable, x_0: torch.Tensor, betas: torch.Tensor, T: int, device=None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x_0 = x_0.to(device)
    betas = betas.to(device)

    batch_size = x_0.shape[0]
    t = torch.randint(1, T + 1, size=(batch_size,), device=device)

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    a_bar_t = alpha_bars[t - 1]

    broadcast_shape = [batch_size] + [1] * (x_0.ndim - 1)
    a_bar_t = a_bar_t.reshape(broadcast_shape)

    epsilon = torch.randn_like(x_0)
    x_t = torch.sqrt(a_bar_t) * x_0 + torch.sqrt(1.0 - a_bar_t) * epsilon

    epsilon_pred = model_predict(x_t, t)
    loss = torch.mean((epsilon - epsilon_pred) ** 2)
    return float(loss.item())
