mutable struct BatchNorm
    eps::Float64
    momentum::Float64
    gamma
    beta
    running_mean
    running_var
end

function BatchNorm(num_features::Int; eps::Float64=1e-5, momentum::Float64=0.1)
    gamma = ones(num_features)
    beta = zeros(num_features)
    running_mean = zeros(num_features)
    running_var = ones(num_features)
    BatchNorm(eps, momentum, gamma, beta, running_mean, running_var)
end

function forward(bn::BatchNorm, x; training::Bool=true)
    original_shape = size(x)
    if length(original_shape) > 2
        batch = original_shape[1]
        channels = original_shape[2]
        x_reshaped = reshape(x, batch, channels, :)
        x_reshaped = reshape(permutedims(x_reshaped, (1, 3, 2)), :, channels)
    else
        x_reshaped = x
        channels = original_shape[end]
    end

    if training
        batch_mean = mean(x_reshaped, dims=1)
        batch_var = var(x_reshaped, dims=1)
        bn.running_mean = (1 - bn.momentum) .* bn.running_mean .+ bn.momentum .* vec(batch_mean)
        bn.running_var = (1 - bn.momentum) .* bn.running_var .+ bn.momentum .* vec(batch_var)
        x_norm = (x_reshaped .- batch_mean) ./ sqrt.(batch_var .+ bn.eps)
    else
        x_norm = (x_reshaped .- bn.running_mean') ./ sqrt.(bn.running_var' .+ bn.eps)
    end

    out = bn.gamma' .* x_norm .+ bn.beta'

    if length(original_shape) > 2
        out = reshape(out, batch, :, channels)
        out = permutedims(out, (1, 3, 2))
        out = reshape(out, original_shape)
    else
        out = reshape(out, original_shape)
    end

    out
end

relu(x) = max.(0, x)

function post_activation_block(x, W1, W2, bn1::BatchNorm, bn2::BatchNorm)
    out = x * W1
    out = forward(bn1, out)
    out = relu(out)
    out = out * W2
    out = forward(bn2, out)
    relu(out .+ x)
end

function pre_activation_block(x, W1, W2, bn1::BatchNorm, bn2::BatchNorm)
    out = forward(bn1, x)
    out = relu(out)
    out = out * W1
    out = forward(bn2, out)
    out = relu(out)
    out = out * W2
    out .+ x
end
