import mlx.core as mx


def compute_ddpm_loss(model_predict: callable, x_0: mx.array, betas: mx.array, T: int) -> float:
    batch_size = x_0.shape[0]
    t = mx.random.randint(1, T + 1, shape=(batch_size,))

    alphas = 1.0 - betas
    alpha_bars = mx.cumprod(alphas, axis=0)
    a_bar_t = alpha_bars[t - 1]

    broadcast_shape = [batch_size] + [1] * (x_0.ndim - 1)
    a_bar_t = mx.reshape(a_bar_t, broadcast_shape)

    epsilon = mx.random.normal(shape=x_0.shape)
    x_t = mx.sqrt(a_bar_t) * x_0 + mx.sqrt(1.0 - a_bar_t) * epsilon

    epsilon_pred = model_predict(x_t, t)
    loss = mx.mean((epsilon - epsilon_pred) ** 2)
    return float(loss.item())
