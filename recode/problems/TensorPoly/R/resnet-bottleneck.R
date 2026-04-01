relu <- function(x) {
  pmax(0, x)
}

BottleneckBlock <- setRefClass(
  "BottleneckBlock",
  fields = list(
    in_ch = "numeric",
    bn_ch = "numeric",
    out_ch = "numeric",
    W1 = "matrix",
    W2 = "matrix",
    W3 = "matrix",
    Ws = "matrix"
  ),
  methods = list(
    initialize = function(in_channels, bottleneck_channels, out_channels) {
      in_ch <<- in_channels
      bn_ch <<- bottleneck_channels
      out_ch <<- out_channels
      W1 <<- matrix(rnorm(in_channels * bottleneck_channels, sd = 0.01), nrow = in_channels, ncol = bottleneck_channels)
      W2 <<- matrix(rnorm(bottleneck_channels * bottleneck_channels, sd = 0.01), nrow = bottleneck_channels, ncol = bottleneck_channels)
      W3 <<- matrix(rnorm(bottleneck_channels * out_channels, sd = 0.01), nrow = bottleneck_channels, ncol = out_channels)
      Ws <<- if (in_channels != out_channels) matrix(rnorm(in_channels * out_channels, sd = 0.01), nrow = in_channels, ncol = out_channels) else NULL
    },
    forward = function(x) {
      identity <- x
      out <- relu(x %*% W1)
      out <- relu(out %*% W2)
      out <- out %*% W3
      if (!is.null(Ws)) {
        identity <- identity %*% Ws
      }
      relu(out + identity)
    }
  )
)
