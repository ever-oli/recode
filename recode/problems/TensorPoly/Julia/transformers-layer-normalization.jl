function layer_norm(x, gamma, beta; eps=1e-6)
    mean_vals = mean(x, dims=ndims(x))
    var_vals = var(x, dims=ndims(x))
    x_normalized = (x .- mean_vals) ./ sqrt.(var_vals .+ eps)
    gamma .* x_normalized .+ beta
end
