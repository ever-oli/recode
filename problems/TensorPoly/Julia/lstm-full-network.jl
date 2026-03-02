sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

mutable struct LSTM
    hidden_dim::Int
    W_f
    W_i
    W_c
    W_o
    b_f
    b_i
    b_c
    b_o
    W_y
    b_y
end

function LSTM(input_dim::Int, hidden_dim::Int, output_dim::Int)
    scale = sqrt(2.0 / (input_dim + hidden_dim))
    W_f = randn(hidden_dim, hidden_dim + input_dim) .* scale
    W_i = randn(hidden_dim, hidden_dim + input_dim) .* scale
    W_c = randn(hidden_dim, hidden_dim + input_dim) .* scale
    W_o = randn(hidden_dim, hidden_dim + input_dim) .* scale
    b_f = zeros(hidden_dim)
    b_i = zeros(hidden_dim)
    b_c = zeros(hidden_dim)
    b_o = zeros(hidden_dim)

    W_y = randn(output_dim, hidden_dim) .* sqrt(2.0 / (hidden_dim + output_dim))
    b_y = zeros(output_dim)

    LSTM(hidden_dim, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o, W_y, b_y)
end

function forward(model::LSTM, X)
    batch_size, seq_len, _ = size(X)
    h_t = zeros(batch_size, model.hidden_dim)
    c_t = zeros(batch_size, model.hidden_dim)
    h_states = Vector{Any}(undef, seq_len)

    for t in 1:seq_len
        x_t = X[:, t, :]
        concat = hcat(h_t, x_t)

        f_t = sigmoid(concat * model.W_f' .+ model.b_f)
        i_t = sigmoid(concat * model.W_i' .+ model.b_i)
        c_tilde = tanh.(concat * model.W_c' .+ model.b_c)
        o_t = sigmoid(concat * model.W_o' .+ model.b_o)

        c_t = f_t .* c_t .+ i_t .* c_tilde
        h_t = o_t .* tanh.(c_t)
        h_states[t] = h_t
    end

    h_all = cat(h_states...; dims=2)
    h_flat = reshape(h_all, :, model.hidden_dim)
    y_flat = h_flat * model.W_y' .+ model.b_y
    y = reshape(y_flat, batch_size, seq_len, :)

    return (y = y, h_last = h_t, C_last = c_t)
end
