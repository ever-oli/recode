sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function output_gate(h_prev, x_t, C_t, W_o, b_o)
    concat = hcat(h_prev, x_t)
    o_t = sigmoid(concat * W_o' .+ b_o)
    h_t = o_t .* tanh.(C_t)
    return (o_t = o_t, h_t = h_t)
end
