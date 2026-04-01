sigmoid <- function(x) {
  1 / (1 + exp(-pmin(pmax(x, -500), 500)))
}

GAN <- setRefClass(
  "GAN",
  fields = list(
    data_dim = "numeric",
    noise_dim = "numeric",
    G_W1 = "matrix",
    G_b1 = "numeric",
    G_W2 = "matrix",
    G_b2 = "numeric",
    D_W1 = "matrix",
    D_b1 = "numeric",
    D_W2 = "matrix",
    D_b2 = "numeric",
    D_W3 = "matrix",
    D_b3 = "numeric",
    d_lr = "numeric",
    g_lr = "numeric"
  ),
  methods = list(
    initialize = function(data_dim, noise_dim) {
      data_dim <<- data_dim
      noise_dim <<- noise_dim
      G_W1 <<- matrix(rnorm(noise_dim * 128, sd = 0.02), nrow = noise_dim, ncol = 128)
      G_b1 <<- numeric(128)
      G_W2 <<- matrix(rnorm(128 * data_dim, sd = 0.02), nrow = 128, ncol = data_dim)
      G_b2 <<- numeric(data_dim)

      D_W1 <<- matrix(rnorm(data_dim * 256, sd = 0.02), nrow = data_dim, ncol = 256)
      D_b1 <<- numeric(256)
      D_W2 <<- matrix(rnorm(256 * 128, sd = 0.02), nrow = 256, ncol = 128)
      D_b2 <<- numeric(128)
      D_W3 <<- matrix(rnorm(128 * 1, sd = 0.02), nrow = 128, ncol = 1)
      D_b3 <<- numeric(1)

      d_lr <<- 0.001
      g_lr <<- 0.001
    },
    .generator_forward = function(z) {
      h <- pmax(0, z %*% G_W1 + G_b1)
      tanh(h %*% G_W2 + G_b2)
    },
    .discriminator_forward = function(x) {
      h1 <- pmax(0.2 * (x %*% D_W1 + D_b1), x %*% D_W1 + D_b1)
      h2 <- pmax(0.2 * (h1 %*% D_W2 + D_b2), h1 %*% D_W2 + D_b2)
      logits <- h2 %*% D_W3 + D_b3
      as.vector(sigmoid(logits))
    },
    generate = function(n) {
      z <- matrix(rnorm(n * noise_dim), nrow = n, ncol = noise_dim)
      .generator_forward(z)
    },
    discriminate = function(x) {
      .discriminator_forward(x)
    },
    train_step = function(real_data) {
      batch_size <- nrow(real_data)
      eps <- 1e-8
      fake_data <- generate(batch_size)
      real_probs <- discriminate(real_data)
      fake_probs <- discriminate(fake_data)
      d_loss <- -mean(log(real_probs + eps) + log(1.0 - fake_probs + eps))
      g_loss <- -mean(log(fake_probs + eps))
      list(d_loss = d_loss, g_loss = g_loss)
    }
  )
)
