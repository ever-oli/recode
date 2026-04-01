function candidate_hidden(h_prev, x_t, r_t, W_h, b_h)
    gated_h = r_t .* h_prev
    concat = hcat(gated_h, x_t)
    linear_transform = concat * W_h' .+ b_h
    tanh.(linear_transform)
end
