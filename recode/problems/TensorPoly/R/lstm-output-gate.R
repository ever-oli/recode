sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

output_gate <- function(h_prev, x_t, C_t, W_o, b_o) {
  concat <- cbind(h_prev, x_t)
  o_t <- sigmoid(concat %*% t(W_o) + b_o)
  h_t <- o_t * tanh(C_t)
  list(o_t = o_t, h_t = h_t)
}
