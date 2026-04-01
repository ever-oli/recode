import torch


def vae_loss(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> dict:
    recon_loss_per_sample = torch.sum((x - x_recon) ** 2, dim=1)
    recon_loss = torch.mean(recon_loss_per_sample)

    var = torch.exp(log_var)
    kl_per_sample = -0.5 * torch.sum(1 + log_var - mu ** 2 - var, dim=1)
    kl_loss = torch.mean(kl_per_sample)

    total_loss = recon_loss + kl_loss
    return {
        "total": float(total_loss.item()),
        "recon": float(recon_loss.item()),
        "kl": float(kl_loss.item()),
    }
