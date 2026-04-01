sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

lstm_cell <- function(x_t, h_prev, C_prev, W_f, W_i, W_c, W_o, b_f, b_i, b_c, b_o) {
  concat <- cbind(h_prev, x_t)
  f_t <- sigmoid(concat %*% t(W_f) + b_f)
  i_t <- sigmoid(concat %*% t(W_i) + b_i)
  c_tilde <- tanh(concat %*% t(W_c) + b_c)
  o_t <- sigmoid(concat %*% t(W_o) + b_o)

  C_t <- f_t * C_prev + i_t * c_tilde
  h_t <- o_t * tanh(C_t)
  list(h_t = h_t, C_t = C_t)
}
