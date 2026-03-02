function rnn_cell(x_t, h_prev, W_xh, W_hh, b_h)
    input_term = x_t * W_xh'
    hidden_term = h_prev * W_hh'
    tanh.(input_term .+ hidden_term .+ b_h)
end
