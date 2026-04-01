sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function update_gate(h_prev, x_t, W_z, b_z)
    concat = hcat(h_prev, x_t)
    linear_transform = concat * W_z' .+ b_z
    sigmoid(linear_transform)
end
