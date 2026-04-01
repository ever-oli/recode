relu <- function(x) {
  pmax(0, x)
}

IdentityBlock <- setRefClass(
  "IdentityBlock",
  fields = list(
    channels = "numeric",
    W1 = "matrix",
    W2 = "matrix"
  ),
  methods = list(
    initialize = function(channels) {
      channels <<- channels
      W1 <<- matrix(rnorm(channels * channels, sd = 0.01), nrow = channels, ncol = channels)
      W2 <<- matrix(rnorm(channels * channels, sd = 0.01), nrow = channels, ncol = channels)
    },
    forward = function(x) {
      identity <- x
      out <- relu(x %*% W1)
      out <- out %*% W2
      out + identity
    }
  )
)
