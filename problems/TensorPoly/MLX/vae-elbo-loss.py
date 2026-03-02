import mlx.core as mx


def vae_loss(x: mx.array, x_recon: mx.array, mu: mx.array, log_var: mx.array) -> dict:
    recon_loss_per_sample = mx.sum(mx.square(x - x_recon), axis=1)
    recon_loss = mx.mean(recon_loss_per_sample)

    var = mx.exp(log_var)
    kl_per_sample = -0.5 * mx.sum(1 + log_var - mx.square(mu) - var, axis=1)
    kl_loss = mx.mean(kl_per_sample)

    total_loss = recon_loss + kl_loss
    return {
        "total": float(total_loss.item()),
        "recon": float(recon_loss.item()),
        "kl": float(kl_loss.item()),
    }
