import mlx.core as mx


def kl_divergence(mu: mx.array, log_var: mx.array) -> float:
    var = mx.exp(log_var)
    kl_element = 1 + log_var - mx.square(mu) - var
    batch_kl = -0.5 * mx.sum(kl_element, axis=1)
    return float(mx.mean(batch_kl).item())
