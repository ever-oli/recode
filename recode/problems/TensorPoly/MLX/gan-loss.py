import mlx.core as mx


def discriminator_loss(real_probs: mx.array, fake_probs: mx.array) -> float:
    eps = 1e-8
    real_probs = mx.clip(real_probs, eps, 1 - eps)
    fake_probs = mx.clip(fake_probs, eps, 1 - eps)

    real_loss = -mx.log(real_probs)
    fake_loss = -mx.log(1 - fake_probs)
    total_loss = mx.mean(real_loss + fake_loss)
    return float(total_loss.item())


def generator_loss(fake_probs: mx.array) -> float:
    eps = 1e-8
    fake_probs = mx.clip(fake_probs, eps, 1 - eps)
    loss = -mx.log(fake_probs)
    return float(mx.mean(loss).item())
