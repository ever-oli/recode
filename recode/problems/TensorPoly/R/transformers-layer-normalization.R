layer_norm <- function(x, gamma, beta, eps = 1e-6) {
  dims <- dim(x)
  keep_axes <- seq_len(length(dims) - 1)
  mean_vals <- apply(x, keep_axes, mean)
  var_vals <- apply(x, keep_axes, var)
  mean_arr <- array(mean_vals, dim = c(dims[-length(dims)], 1))
  var_arr <- array(var_vals, dim = c(dims[-length(dims)], 1))
  x_normalized <- (x - mean_arr) / sqrt(var_arr + eps)
  gamma * x_normalized + beta
}
