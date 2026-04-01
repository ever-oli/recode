sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function reset_gate(h_prev, x_t, W_r, b_r)
    concat = hcat(h_prev, x_t)
    linear_transform = concat * W_r' .+ b_r
    sigmoid(linear_transform)
end
