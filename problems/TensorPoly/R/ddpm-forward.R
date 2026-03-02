get_alpha_bar <- function(betas) {
  alphas <- 1.0 - betas
  cumprod(alphas)
}

forward_diffusion <- function(x_0, t, betas) {
  alpha_bar <- get_alpha_bar(betas)
  alpha_bar_t <- alpha_bar[t]

  epsilon <- array(rnorm(length(x_0)), dim = dim(x_0))

  sqrt_alpha_bar_t <- sqrt(alpha_bar_t)
  sqrt_one_minus_alpha_bar_t <- sqrt(1.0 - alpha_bar_t)

  x_t <- sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
  list(x_t = x_t, epsilon = epsilon)
}
