compute_ddpm_loss <- function(model_predict, x_0, betas, T) {
  batch_size <- dim(x_0)[1]
  t <- sample(1:T, batch_size, replace = TRUE)

  alphas <- 1.0 - betas
  alpha_bars <- cumprod(alphas)
  a_bar_t <- alpha_bars[t]

  broadcast_shape <- c(length(a_bar_t), rep(1, length(dim(x_0)) - 1))
  a_bar_t <- array(a_bar_t, dim = broadcast_shape)

  epsilon <- array(rnorm(length(x_0)), dim = dim(x_0))
  x_t <- sqrt(a_bar_t) * x_0 + sqrt(1.0 - a_bar_t) * epsilon

  epsilon_pred <- model_predict(x_t, t)
  loss <- mean((epsilon - epsilon_pred) ^ 2)
  as.numeric(loss)
}
