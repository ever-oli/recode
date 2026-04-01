sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

GRU <- setRefClass(
  "GRU",
  fields = list(
    hidden_dim = "numeric",
    W_r = "matrix",
    W_z = "matrix",
    W_h = "matrix",
    b_r = "numeric",
    b_z = "numeric",
    b_h = "numeric",
    W_y = "matrix",
    b_y = "numeric"
  ),
  methods = list(
    initialize = function(input_dim, hidden_dim, output_dim) {
      hidden_dim <<- hidden_dim
      scale <- sqrt(2.0 / (input_dim + hidden_dim))

      W_r <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      W_z <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      W_h <<- matrix(rnorm(hidden_dim * (hidden_dim + input_dim)), nrow = hidden_dim) * scale
      b_r <<- numeric(hidden_dim)
      b_z <<- numeric(hidden_dim)
      b_h <<- numeric(hidden_dim)

      W_y <<- matrix(rnorm(output_dim * hidden_dim), nrow = output_dim) * sqrt(2.0 / (hidden_dim + output_dim))
      b_y <<- numeric(output_dim)
    },
    forward = function(X) {
      dims <- dim(X)
      batch_size <- dims[1]
      seq_len <- dims[2]
      h_t <- matrix(0, nrow = batch_size, ncol = hidden_dim)

      h_states <- list()
      for (t in seq_len(seq_len)) {
        x_t <- X[, t, ]
        concat <- cbind(h_t, x_t)
        r_t <- sigmoid(concat %*% t(W_r) + b_r)
        z_t <- sigmoid(concat %*% t(W_z) + b_z)

        gated_h <- r_t * h_t
        concat_cand <- cbind(gated_h, x_t)
        h_tilde <- tanh(concat_cand %*% t(W_h) + b_h)

        h_t <- z_t * h_t + (1 - z_t) * h_tilde
        h_states[[t]] <- h_t
      }

      h_all <- array(unlist(h_states), dim = c(batch_size, seq_len, hidden_dim))
      h_flat <- matrix(h_all, ncol = hidden_dim)
      y_flat <- h_flat %*% t(W_y) + b_y
      y <- array(y_flat, dim = c(batch_size, seq_len, nrow(W_y)))

      list(y = y, h_last = h_t)
    }
  )
)
