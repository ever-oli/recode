function reparameterize(mu, log_var)
    std = exp.(0.5 .* log_var)
    epsilon = randn(size(mu))
    mu .+ std .* epsilon
end
