rnn_forward <- function(X, h_0, W_xh, W_hh, b_h) {
  dims <- dim(X)
  batch_size <- dims[1]
  time_steps <- dims[2]

  h_all_list <- list()
  h_current <- h_0

  for (t in seq_len(time_steps)) {
    x_t <- X[, t, ]
    h_current <- tanh(x_t %*% t(W_xh) + h_current %*% t(W_hh) + b_h)
    h_all_list[[t]] <- h_current
  }

  h_all <- array(unlist(h_all_list), dim = c(batch_size, time_steps, ncol(h_current)))
  list(h_all = h_all, h_final = h_current)
}
