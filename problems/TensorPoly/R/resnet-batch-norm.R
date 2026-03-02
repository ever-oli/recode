BatchNorm <- setRefClass(
  "BatchNorm",
  fields = list(
    eps = "numeric",
    momentum = "numeric",
    gamma = "numeric",
    beta = "numeric",
    running_mean = "numeric",
    running_var = "numeric"
  ),
  methods = list(
    initialize = function(num_features, eps = 1e-5, momentum = 0.1) {
      eps <<- eps
      momentum <<- momentum
      gamma <<- rep(1, num_features)
      beta <<- rep(0, num_features)
      running_mean <<- rep(0, num_features)
      running_var <<- rep(1, num_features)
    },
    forward = function(x, training = TRUE) {
      original_shape <- dim(x)
      if (length(original_shape) > 2) {
        batch <- original_shape[1]
        channels <- original_shape[2]
        x_reshaped <- array(x, dim = c(batch, channels, prod(original_shape[-c(1, 2)])))
        x_reshaped <- array(aperm(x_reshaped, c(1, 3, 2)), dim = c(-1, channels))
      } else {
        x_reshaped <- x
        channels <- original_shape[length(original_shape)]
      }

      if (training) {
        batch_mean <- colMeans(x_reshaped)
        batch_var <- apply(x_reshaped, 2, var)
        running_mean <<- (1 - momentum) * running_mean + momentum * batch_mean
        running_var <<- (1 - momentum) * running_var + momentum * batch_var
        x_norm <- (x_reshaped - batch_mean) / sqrt(batch_var + eps)
      } else {
        x_norm <- (x_reshaped - running_mean) / sqrt(running_var + eps)
      }

      out <- gamma * x_norm + beta

      if (length(original_shape) > 2) {
        out <- array(out, dim = c(original_shape[1], prod(original_shape[-c(1, 2)]), channels))
        out <- aperm(out, c(1, 3, 2))
        out <- array(out, dim = original_shape)
      } else {
        out <- array(out, dim = original_shape)
      }

      out
    }
  )
)

relu <- function(x) {
  pmax(0, x)
}

post_activation_block <- function(x, W1, W2, bn1, bn2) {
  out <- x %*% W1
  out <- bn1$forward(out)
  out <- relu(out)
  out <- out %*% W2
  out <- bn2$forward(out)
  relu(out + x)
}

pre_activation_block <- function(x, W1, W2, bn1, bn2) {
  out <- bn1$forward(x)
  out <- relu(out)
  out <- out %*% W1
  out <- bn2$forward(out)
  out <- relu(out)
  out <- out %*% W2
  out + x
}
