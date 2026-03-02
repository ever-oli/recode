import mlx.core as mx


def reparameterize(mu: mx.array, log_var: mx.array) -> mx.array:
    std = mx.exp(0.5 * log_var)
    epsilon = mx.random.normal(shape=mu.shape)
    return mu + std * epsilon
