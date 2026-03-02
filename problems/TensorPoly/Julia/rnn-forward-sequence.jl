function rnn_forward(X, h_0, W_xh, W_hh, b_h)
    batch_size, time_steps, _ = size(X)
    h_current = h_0
    h_all_list = Vector{Any}(undef, time_steps)

    for t in 1:time_steps
        x_t = X[:, t, :]
        h_current = tanh.(x_t * W_xh' .+ h_current * W_hh' .+ b_h)
        h_all_list[t] = h_current
    end

    h_all = cat(h_all_list...; dims=2)
    return (h_all = h_all, h_final = h_current)
end
