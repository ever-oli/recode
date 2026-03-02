sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

input_gate <- function(h_prev, x_t, W_i, b_i, W_c, b_c) {
  concat <- cbind(h_prev, x_t)
  i_t <- sigmoid(concat %*% t(W_i) + b_i)
  c_tilde <- tanh(concat %*% t(W_c) + b_c)
  list(i_t = i_t, c_tilde = c_tilde)
}
