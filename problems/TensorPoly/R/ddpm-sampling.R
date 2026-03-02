ddpm_sample <- function(model_predict, shape, betas, T) {
  x_t <- array(rnorm(prod(shape)), dim = shape)

  alphas <- 1.0 - betas
  alpha_bars <- cumprod(alphas)

  for (t in T:1) {
    epsilon_pred <- model_predict(x_t, t)

    beta_t <- betas[t]
    alpha_t <- alphas[t]
    alpha_bar_t <- alpha_bars[t]

    inv_sqrt_alpha_t <- 1.0 / sqrt(alpha_t)
    noise_coeff <- beta_t / sqrt(1.0 - alpha_bar_t)

    mu <- inv_sqrt_alpha_t * (x_t - noise_coeff * epsilon_pred)

    if (t > 1) {
      sigma_t <- sqrt(beta_t)
      z <- array(rnorm(prod(shape)), dim = shape)
      x_t <- mu + sigma_t * z
    } else {
      x_t <- mu
    }
  }

  x_t
}
