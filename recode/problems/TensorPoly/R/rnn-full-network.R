VanillaRNN <- setRefClass(
  "VanillaRNN",
  fields = list(
    hidden_dim = "numeric",
    W_xh = "matrix",
    W_hh = "matrix",
    W_hy = "matrix",
    b_h = "numeric",
    b_y = "numeric"
  ),
  methods = list(
    initialize = function(input_dim, hidden_dim, output_dim) {
      hidden_dim <<- hidden_dim
      W_xh <<- matrix(rnorm(hidden_dim * input_dim), nrow = hidden_dim) * sqrt(2.0 / (input_dim + hidden_dim))
      W_hh <<- matrix(rnorm(hidden_dim * hidden_dim), nrow = hidden_dim) * sqrt(2.0 / (2 * hidden_dim))
      W_hy <<- matrix(rnorm(output_dim * hidden_dim), nrow = output_dim) * sqrt(2.0 / (hidden_dim + output_dim))
      b_h <<- numeric(hidden_dim)
      b_y <<- numeric(output_dim)
    },
    forward = function(X, h_0 = NULL) {
      dims <- dim(X)
      batch_size <- dims[1]
      time_steps <- dims[2]

      if (is.null(h_0)) {
        h_current <- matrix(0, nrow = batch_size, ncol = hidden_dim)
      } else {
        h_current <- h_0
      }

      h_list <- list()
      for (t in seq_len(time_steps)) {
        x_t <- X[, t, ]
        h_current <- tanh(x_t %*% t(W_xh) + h_current %*% t(W_hh) + b_h)
        h_list[[t]] <- h_current
      }

      h_seq <- array(unlist(h_list), dim = c(batch_size, time_steps, hidden_dim))
      h_final <- h_current

      h_flat <- matrix(h_seq, ncol = hidden_dim)
      y_flat <- h_flat %*% t(W_hy) + b_y
      y_seq <- array(y_flat, dim = c(batch_size, time_steps, nrow(W_hy)))

      list(y_seq = y_seq, h_final = h_final)
    }
  )
)
