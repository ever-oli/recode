relu <- function(x) {
  pmax(0, x)
}

ConvBlock <- setRefClass(
  "ConvBlock",
  fields = list(
    in_channels = "numeric",
    out_channels = "numeric",
    W1 = "matrix",
    W2 = "matrix",
    Ws = "matrix"
  ),
  methods = list(
    initialize = function(in_channels, out_channels) {
      in_channels <<- in_channels
      out_channels <<- out_channels
      W1 <<- matrix(rnorm(in_channels * out_channels, sd = 0.01), nrow = in_channels, ncol = out_channels)
      W2 <<- matrix(rnorm(out_channels * out_channels, sd = 0.01), nrow = out_channels, ncol = out_channels)
      Ws <<- matrix(rnorm(in_channels * out_channels, sd = 0.01), nrow = in_channels, ncol = out_channels)
    },
    forward = function(x) {
      main <- relu(x %*% W1)
      main <- main %*% W2
      shortcut <- x %*% Ws
      relu(main + shortcut)
    }
  )
)
