mutable struct VanillaRNN
    hidden_dim::Int
    W_xh
    W_hh
    W_hy
    b_h
    b_y
end

function VanillaRNN(input_dim::Int, hidden_dim::Int, output_dim::Int)
    W_xh = randn(hidden_dim, input_dim) .* sqrt(2.0 / (input_dim + hidden_dim))
    W_hh = randn(hidden_dim, hidden_dim) .* sqrt(2.0 / (2 * hidden_dim))
    W_hy = randn(output_dim, hidden_dim) .* sqrt(2.0 / (hidden_dim + output_dim))
    b_h = zeros(hidden_dim)
    b_y = zeros(output_dim)
    VanillaRNN(hidden_dim, W_xh, W_hh, W_hy, b_h, b_y)
end

function forward(model::VanillaRNN, X, h_0=nothing)
    batch_size, time_steps, _ = size(X)
    h_current = h_0 === nothing ? zeros(batch_size, model.hidden_dim) : h_0
    h_list = Vector{Any}(undef, time_steps)

    for t in 1:time_steps
        x_t = X[:, t, :]
        h_current = tanh.(x_t * model.W_xh' .+ h_current * model.W_hh' .+ model.b_h)
        h_list[t] = h_current
    end

    h_seq = cat(h_list...; dims=2)
    h_final = h_current

    h_flat = reshape(h_seq, :, model.hidden_dim)
    y_flat = h_flat * model.W_hy' .+ model.b_y
    y_seq = reshape(y_flat, batch_size, time_steps, :)

    return (y_seq = y_seq, h_final = h_final)
end
