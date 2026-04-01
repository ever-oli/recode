kl_divergence <- function(mu, log_var) {
  var <- exp(log_var)
  kl_element <- 1 + log_var - (mu ^ 2) - var
  batch_kl <- -0.5 * rowSums(kl_element)
  mean(batch_kl)
}
