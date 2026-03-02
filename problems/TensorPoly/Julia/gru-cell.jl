sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function gru_cell(x_t, h_prev, W_r, W_z, W_h, b_r, b_z, b_h)
    concat_gates = hcat(h_prev, x_t)
    r_t = sigmoid(concat_gates * W_r' .+ b_r)
    z_t = sigmoid(concat_gates * W_z' .+ b_z)

    gated_h = r_t .* h_prev
    concat_cand = hcat(gated_h, x_t)
    h_tilde = tanh.(concat_cand * W_h' .+ b_h)

    z_t .* h_prev .+ (1 .- z_t) .* h_tilde
end
