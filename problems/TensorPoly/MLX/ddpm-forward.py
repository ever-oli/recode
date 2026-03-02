import mlx.core as mx


def get_alpha_bar(betas: mx.array) -> mx.array:
    alphas = 1.0 - betas
    return mx.cumprod(alphas, axis=0)


def forward_diffusion(x_0: mx.array, t: int, betas: mx.array) -> tuple:
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]

    epsilon = mx.random.normal(shape=x_0.shape)

    sqrt_alpha_bar_t = mx.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = mx.sqrt(1.0 - alpha_bar_t)

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    return x_t, epsilon
