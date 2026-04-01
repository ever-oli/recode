discriminator_loss <- function(real_probs, fake_probs) {
  eps <- 1e-8
  real_probs <- pmin(pmax(real_probs, eps), 1 - eps)
  fake_probs <- pmin(pmax(fake_probs, eps), 1 - eps)
  real_loss <- -log(real_probs)
  fake_loss <- -log(1 - fake_probs)
  mean(real_loss + fake_loss)
}

generator_loss <- function(fake_probs) {
  eps <- 1e-8
  fake_probs <- pmin(pmax(fake_probs, eps), 1 - eps)
  loss <- -log(fake_probs)
  mean(loss)
}
