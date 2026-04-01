sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

reset_gate <- function(h_prev, x_t, W_r, b_r) {
  concat <- cbind(h_prev, x_t)
  linear_transform <- concat %*% t(W_r) + b_r
  sigmoid(linear_transform)
}
