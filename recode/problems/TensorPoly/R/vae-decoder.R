vae_decoder <- function(z, output_dim) {
  dims <- dim(z)
  latent_dim <- dims[2]
  hidden_dim <- 256

  w_h <- matrix(rnorm(latent_dim * hidden_dim, sd = 0.01), nrow = latent_dim, ncol = hidden_dim)
  b_h <- numeric(hidden_dim)
  h <- pmax(0, z %*% w_h + b_h)

  w_out <- matrix(rnorm(hidden_dim * output_dim, sd = 0.01), nrow = hidden_dim, ncol = output_dim)
  b_out <- numeric(output_dim)
  logits <- h %*% w_out + b_out

  1 / (1 + exp(-logits))
}
