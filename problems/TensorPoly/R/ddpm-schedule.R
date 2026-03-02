linear_beta_schedule <- function(T, beta_1 = 0.0001, beta_T = 0.02) {
  seq(beta_1, beta_T, length.out = T)
}

cosine_alpha_bar_schedule <- function(T, s = 0.008) {
  t <- 1:T
  f_0 <- cos(s / (1 + s) * pi / 2) ^ 2
  f_t <- cos(((t / T) + s) / (1 + s) * pi / 2) ^ 2
  f_t / f_0
}

alpha_bar_to_betas <- function(alpha_bars) {
  alpha_bars_prev <- c(1.0, alpha_bars[-length(alpha_bars)])
  betas <- 1.0 - (alpha_bars / alpha_bars_prev)
  pmin(pmax(betas, 0.0), 0.999)
}
