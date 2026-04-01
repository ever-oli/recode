function get_alpha_bar(betas)
    alphas = 1.0 .- betas
    cumprod(alphas)
end

function forward_diffusion(x_0, t::Int, betas)
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t]

    epsilon = randn(size(x_0))

    sqrt_alpha_bar_t = sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = sqrt(1.0 - alpha_bar_t)

    x_t = sqrt_alpha_bar_t .* x_0 .+ sqrt_one_minus_alpha_bar_t .* epsilon
    return (x_t = x_t, epsilon = epsilon)
end
