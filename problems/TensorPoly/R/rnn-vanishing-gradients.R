compute_gradient_norm_decay <- function(T, W_hh) {
  spectral_norm <- norm(W_hh, type = "2")
  norms <- numeric(T)
  norms[1] <- 1.0
  current_norm <- 1.0

  if (T > 1) {
    for (i in 2:T) {
      current_norm <- current_norm * spectral_norm
      norms[i] <- current_norm
    }
  }

  norms
}
