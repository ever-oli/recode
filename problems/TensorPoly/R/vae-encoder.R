vae_encoder <- function(x, latent_dim) {
  dims <- dim(x)
  input_dim <- dims[2]
  hidden_dim <- 256

  w_h <- matrix(rnorm(input_dim * hidden_dim, sd = 0.01), nrow = input_dim, ncol = hidden_dim)
  b_h <- numeric(hidden_dim)
  h <- pmax(0, x %*% w_h + b_h)

  w_mu <- matrix(rnorm(hidden_dim * latent_dim, sd = 0.01), nrow = hidden_dim, ncol = latent_dim)
  b_mu <- numeric(latent_dim)
  mu <- h %*% w_mu + b_mu

  w_log_var <- matrix(rnorm(hidden_dim * latent_dim, sd = 0.01), nrow = hidden_dim, ncol = latent_dim)
  b_log_var <- numeric(latent_dim)
  log_var <- h %*% w_log_var + b_log_var

  list(mu = mu, log_var = log_var)
}
