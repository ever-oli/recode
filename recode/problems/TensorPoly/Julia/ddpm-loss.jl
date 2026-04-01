function compute_ddpm_loss(model_predict, x_0, betas, T::Int)
    batch_size = size(x_0, 1)
    t = rand(1:T, batch_size)

    alphas = 1.0 .- betas
    alpha_bars = cumprod(alphas)
    a_bar_t = alpha_bars[t]

    broadcast_shape = (batch_size, ones(Int, ndims(x_0) - 1)...)
    a_bar_t = reshape(a_bar_t, broadcast_shape)

    epsilon = randn(size(x_0))
    x_t = sqrt.(a_bar_t) .* x_0 .+ sqrt.(1.0 .- a_bar_t) .* epsilon

    epsilon_pred = model_predict(x_t, t)
    loss = mean((epsilon .- epsilon_pred) .^ 2)
    return Float64(loss)
end
