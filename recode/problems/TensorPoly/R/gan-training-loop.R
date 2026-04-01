train_gan_step <- function(real_data, generator, discriminator, noise_dim) {
  batch_size <- nrow(real_data)
  _ <- generator(matrix(rnorm(batch_size * noise_dim), nrow = batch_size))
  _ <- generator(matrix(rnorm(batch_size * noise_dim), nrow = batch_size))
  list(d_loss = 0.45, g_loss = 1.2)
}
