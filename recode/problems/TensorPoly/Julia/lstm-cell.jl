sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function lstm_cell(x_t, h_prev, C_prev, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o)
    concat = hcat(h_prev, x_t)
    f_t = sigmoid(concat * W_f' .+ b_f)
    i_t = sigmoid(concat * W_i' .+ b_i)
    c_tilde = tanh.(concat * W_c' .+ b_c)
    o_t = sigmoid(concat * W_o' .+ b_o)

    C_t = f_t .* C_prev .+ i_t .* c_tilde
    h_t = o_t .* tanh.(C_t)
    return (h_t = h_t, C_t = C_t)
end
