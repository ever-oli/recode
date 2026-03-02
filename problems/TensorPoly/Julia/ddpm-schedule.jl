function linear_beta_schedule(T::Int; beta_1::Float64=0.0001, beta_T::Float64=0.02)
    range(beta_1, beta_T; length=T)
end

function cosine_alpha_bar_schedule(T::Int; s::Float64=0.008)
    t = 1:T
    f_0 = cos(s / (1 + s) * pi / 2) ^ 2
    f_t = cos.(((t ./ T) .+ s) ./ (1 + s) .* (pi / 2)) .^ 2
    f_t ./ f_0
end

function alpha_bar_to_betas(alpha_bars)
    alpha_bars_prev = vcat(1.0, alpha_bars[1:end-1])
    betas = 1.0 .- (alpha_bars ./ alpha_bars_prev)
    clamp.(betas, 0.0, 0.999)
end
