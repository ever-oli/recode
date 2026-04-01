function bptt_single_step(dh_next, h_t, h_prev, x_t, W_hh)
    dtanh = (1 .- h_t .^ 2) .* dh_next
    dW_hh = dtanh' * h_prev
    dh_prev = dtanh * W_hh
    return (dh_prev = dh_prev, dW_hh = dW_hh)
end
