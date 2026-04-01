vae_loss <- function(x, x_recon, mu, log_var) {
  recon_loss_per_sample <- rowSums((x - x_recon) ^ 2)
  recon_loss <- mean(recon_loss_per_sample)

  var <- exp(log_var)
  kl_per_sample <- -0.5 * rowSums(1 + log_var - (mu ^ 2) - var)
  kl_loss <- mean(kl_per_sample)

  total_loss <- recon_loss + kl_loss
  list(total = total_loss, recon = recon_loss, kl = kl_loss)
}
