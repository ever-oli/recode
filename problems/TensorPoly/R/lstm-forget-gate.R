sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

forget_gate <- function(h_prev, x_t, W_f, b_f) {
  concat <- cbind(h_prev, x_t)
  linear_transform <- concat %*% t(W_f) + b_f
  sigmoid(linear_transform)
}
