sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

update_gate <- function(h_prev, x_t, W_z, b_z) {
  concat <- cbind(h_prev, x_t)
  linear_transform <- concat %*% t(W_z) + b_z
  sigmoid(linear_transform)
}
