sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function forget_gate(h_prev, x_t, W_f, b_f)
    concat = hcat(h_prev, x_t)
    linear_transform = concat * W_f' .+ b_f
    sigmoid(linear_transform)
end
