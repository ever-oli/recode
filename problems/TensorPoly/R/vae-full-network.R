VAE <- setRefClass(
  "VAE",
  fields = list(
    input_dim = "numeric",
    latent_dim = "numeric",
    hidden_dim = "numeric",
    w_enc = "matrix",
    b_enc = "numeric",
    w_mu = "matrix",
    b_mu = "numeric",
    w_log_var = "matrix",
    b_log_var = "numeric",
    w_dec_h = "matrix",
    b_dec_h = "numeric",
    w_dec_out = "matrix",
    b_dec_out = "numeric"
  ),
  methods = list(
    initialize = function(input_dim, latent_dim) {
      input_dim <<- input_dim
      latent_dim <<- latent_dim
      hidden_dim <<- 256

      w_enc <<- matrix(rnorm(input_dim * hidden_dim, sd = 0.01), nrow = input_dim, ncol = hidden_dim)
      b_enc <<- numeric(hidden_dim)

      w_mu <<- matrix(rnorm(hidden_dim * latent_dim, sd = 0.01), nrow = hidden_dim, ncol = latent_dim)
      b_mu <<- numeric(latent_dim)
      w_log_var <<- matrix(rnorm(hidden_dim * latent_dim, sd = 0.01), nrow = hidden_dim, ncol = latent_dim)
      b_log_var <<- numeric(latent_dim)

      w_dec_h <<- matrix(rnorm(latent_dim * hidden_dim, sd = 0.01), nrow = latent_dim, ncol = hidden_dim)
      b_dec_h <<- numeric(hidden_dim)
      w_dec_out <<- matrix(rnorm(hidden_dim * input_dim, sd = 0.01), nrow = hidden_dim, ncol = input_dim)
      b_dec_out <<- numeric(input_dim)
    },
    forward = function(x) {
      h_enc <- pmax(0, x %*% w_enc + b_enc)
      mu <- h_enc %*% w_mu + b_mu
      log_var <- h_enc %*% w_log_var + b_log_var

      std <- exp(0.5 * log_var)
      eps <- matrix(rnorm(length(mu)), nrow = nrow(mu), ncol = ncol(mu))
      z <- mu + std * eps

      h_dec <- pmax(0, z %*% w_dec_h + b_dec_h)
      logits <- h_dec %*% w_dec_out + b_dec_out
      x_recon <- 1 / (1 + exp(-logits))

      list(x_recon = x_recon, mu = mu, log_var = log_var)
    },
    generate = function(n_samples) {
      z <- matrix(rnorm(n_samples * latent_dim), nrow = n_samples, ncol = latent_dim)
      h_dec <- pmax(0, z %*% w_dec_h + b_dec_h)
      logits <- h_dec %*% w_dec_out + b_dec_out
      1 / (1 + exp(-logits))
    }
  )
)
