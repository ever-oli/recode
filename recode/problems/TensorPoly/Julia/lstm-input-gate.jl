sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function input_gate(h_prev, x_t, W_i, b_i, W_c, b_c)
    concat = hcat(h_prev, x_t)
    i_t = sigmoid(concat * W_i' .+ b_i)
    c_tilde = tanh.(concat * W_c' .+ b_c)
    return (i_t = i_t, c_tilde = c_tilde)
end
