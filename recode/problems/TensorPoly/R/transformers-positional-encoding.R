positional_encoding <- function(seq_length, d_model) {
  position <- matrix(0:(seq_length - 1), ncol = 1)
  i <- seq(0, d_model - 1, by = 2)
  div_term <- exp(i * (-log(10000.0) / d_model))

  pe <- matrix(0, nrow = seq_length, ncol = d_model)
  sin_idx <- seq(1, d_model, by = 2)
  cos_idx <- seq(2, d_model, by = 2)
  pe[, sin_idx] <- sin(position %*% t(div_term))
  if (length(cos_idx) > 0) {
    pe[, cos_idx] <- cos(position %*% t(div_term[1:length(cos_idx)]))
  }
  pe
}
