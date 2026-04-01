local_response_normalization <- function(x, k = 2, n = 5, alpha = 1e-4, beta = 0.75) {
  dims <- dim(x)
  batch_size <- dims[1]
  h <- dims[2]
  w <- dims[3]
  c <- dims[4]

  squared_x <- x ^ 2
  pad <- n %/% 2
  padded_sq <- array(0, dim = c(batch_size, h, w, c + 2 * pad))
  padded_sq[, , , (pad + 1):(pad + c)] <- squared_x

  sum_sq <- array(0, dim = c(batch_size, h, w, c))
  for (i in seq_len(n)) {
    sum_sq <- sum_sq + padded_sq[, , , i:(i + c - 1)]
  }

  scale <- (k + alpha * sum_sq) ^ beta
  x / scale
}
