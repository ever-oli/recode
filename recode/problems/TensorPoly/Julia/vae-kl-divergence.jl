function kl_divergence(mu, log_var)
    var = exp.(log_var)
    kl_element = 1 .+ log_var .- mu .^ 2 .- var
    batch_kl = -0.5 .* sum(kl_element, dims=2)
    return Float64(mean(batch_kl))
end
