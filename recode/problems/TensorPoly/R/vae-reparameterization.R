reparameterize <- function(mu, log_var) {
  std <- exp(0.5 * log_var)
  epsilon <- matrix(rnorm(length(mu)), nrow = nrow(mu), ncol = ncol(mu))
  mu + std * epsilon
}
