bptt_single_step <- function(dh_next, h_t, h_prev, x_t, W_hh) {
  dtanh <- (1 - (h_t ^ 2)) * dh_next
  dW_hh <- t(dtanh) %*% h_prev
  dh_prev <- dtanh %*% W_hh
  list(dh_prev = dh_prev, dW_hh = dW_hh)
}
