sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

mutable struct GRU
    hidden_dim::Int
    W_r
    W_z
    W_h
    b_r
    b_z
    b_h
    W_y
    b_y
end

function GRU(input_dim::Int, hidden_dim::Int, output_dim::Int)
    scale = sqrt(2.0 / (input_dim + hidden_dim))
    W_r = randn(hidden_dim, hidden_dim + input_dim) .* scale
    W_z = randn(hidden_dim, hidden_dim + input_dim) .* scale
    W_h = randn(hidden_dim, hidden_dim + input_dim) .* scale
    b_r = zeros(hidden_dim)
    b_z = zeros(hidden_dim)
    b_h = zeros(hidden_dim)

    W_y = randn(output_dim, hidden_dim) .* sqrt(2.0 / (hidden_dim + output_dim))
    b_y = zeros(output_dim)

    GRU(hidden_dim, W_r, W_z, W_h, b_r, b_z, b_h, W_y, b_y)
end

function forward(model::GRU, X)
    batch_size, seq_len, _ = size(X)
    h_t = zeros(batch_size, model.hidden_dim)
    h_states = Vector{Any}(undef, seq_len)

    for t in 1:seq_len
        x_t = X[:, t, :]
        concat = hcat(h_t, x_t)
        r_t = sigmoid(concat * model.W_r' .+ model.b_r)
        z_t = sigmoid(concat * model.W_z' .+ model.b_z)

        gated_h = r_t .* h_t
        concat_cand = hcat(gated_h, x_t)
        h_tilde = tanh.(concat_cand * model.W_h' .+ model.b_h)

        h_t = z_t .* h_t .+ (1 .- z_t) .* h_tilde
        h_states[t] = h_t
    end

    h_all = cat(h_states...; dims=2)
    h_flat = reshape(h_all, :, model.hidden_dim)
    y_flat = h_flat * model.W_y' .+ model.b_y
    y = reshape(y_flat, batch_size, seq_len, :)

    return (y = y, h_last = h_t)
end
