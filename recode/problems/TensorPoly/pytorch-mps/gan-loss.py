import torch


def discriminator_loss(real_probs: torch.Tensor, fake_probs: torch.Tensor) -> float:
    eps = 1e-8
    real_probs = torch.clamp(real_probs, eps, 1 - eps)
    fake_probs = torch.clamp(fake_probs, eps, 1 - eps)
    real_loss = -torch.log(real_probs)
    fake_loss = -torch.log(1 - fake_probs)
    total_loss = torch.mean(real_loss + fake_loss)
    return float(total_loss.item())


def generator_loss(fake_probs: torch.Tensor) -> float:
    eps = 1e-8
    fake_probs = torch.clamp(fake_probs, eps, 1 - eps)
    loss = -torch.log(fake_probs)
    return float(torch.mean(loss).item())
