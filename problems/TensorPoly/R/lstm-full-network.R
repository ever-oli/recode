sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

LSTM <- setRefClass(
  "LSTM",
  fields = list(
    hidden_dim = "numeric",
    W_f = "matrix",
    W_i = "matrix",
    W_c = "matrix",
    W_o = "matrix",
    b_f = "numeric",
    b_i = "numeric",
    b_c = "numeric",
    b_o = "numeric",
    W_y = "matrix",
    b_y = "numeric"
  ),
  methods = list(
    initialize = function(input_dim, hidden_dim, output_dim) {
      hidden_dim <<- hidden_dim
      scale <- sqrt(2.0 / (input_dim + hidden_dim))

      W_f <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      W_i <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      W_c <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      W_o <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      b_f <<- numeric(hidden_dim)
      b_i <<- numeric(hidden_dim)
      b_c <<- numeric(hidden_dim)
      b_o <<- numeric(hidden_dim)

      W_y <<- matrix(rnorm(output_dim * hidden_dim), nrow = output_dim) * sqrt(2.0 / (hidden_dim + output_dim))
      b_y <<- numeric(output_dim)
    },
    forward = function(X) {
      dims <- dim(X)
      batch_size <- dims[1]
      seq_len <- dims[2]
      h_t <- matrix(0, nrow = batch_size, ncol = hidden_dim)
      c_t <- matrix(0, nrow = batch_size, ncol = hidden_dim)

      h_states <- list()
      for (t in seq_len(seq_len)) {
        x_t <- X[, t, ]
        concat <- cbind(h_t, x_t)

        f_t <- sigmoid(concat %*% t(W_f) + b_f)
        i_t <- sigmoid(concat %*% t(W_i) + b_i)
        c_tilde <- tanh(concat %*% t(W_c) + b_c)
        o_t <- sigmoid(concat %*% t(W_o) + b_o)

        c_t <- f_t * c_t + i_t * c_tilde
        h_t <- o_t * tanh(c_t)
        h_states[[t]] <- h_t
      }

      h_all <- array(unlist(h_states), dim = c(batch_size, seq_len, hidden_dim))
      h_flat <- matrix(h_all, ncol = hidden_dim)
      y_flat <- h_flat %*% t(W_y) + b_y
      y <- array(y_flat, dim = c(batch_size, seq_len, nrow(W_y)))

      list(y = y, h_last = h_t, C_last = c_t)
    }
  )
)
