generator <- function(z, output_dim) {
  noise_dim <- ncol(z)
  W1 <- matrix(rnorm(noise_dim * 128, sd = 0.02), nrow = noise_dim, ncol = 128)
  b1 <- numeric(128)
  W2 <- matrix(rnorm(128 * output_dim, sd = 0.02), nrow = 128, ncol = output_dim)
  b2 <- numeric(output_dim)

  h1 <- pmax(0, z %*% W1 + b1)
  tanh(h1 %*% W2 + b2)
}
