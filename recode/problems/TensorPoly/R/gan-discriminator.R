sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

discriminator <- function(x) {
  input_dim <- ncol(x)

  W1 <- matrix(rnorm(input_dim * 256, sd = 0.02), nrow = input_dim, ncol = 256)
  b1 <- numeric(256)
  W2 <- matrix(rnorm(256 * 128, sd = 0.02), nrow = 256, ncol = 128)
  b2 <- numeric(128)
  W3 <- matrix(rnorm(128 * 1, sd = 0.02), nrow = 128, ncol = 1)
  b3 <- numeric(1)

  h1 <- x %*% W1 + b1
  h1 <- pmax(0.2 * h1, h1)
  h2 <- h1 %*% W2 + b2
  h2 <- pmax(0.2 * h2, h2)
  logits <- h2 %*% W3 + b3
  sigmoid(logits)
}
