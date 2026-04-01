conv_relu <- function(x, out_channels) {
  C <- dim(x)[4]
  W_weights <- array(rnorm(C * out_channels) * 0.1, dim = c(C, out_channels))
  x <- apply(x, c(1, 2, 3), function(slice) slice %*% W_weights)
  pmax(0, x)
}

maxpool_2x2 <- function(x) {
  B <- dim(x)[1]
  H <- dim(x)[2]
  W <- dim(x)[3]
  C <- dim(x)[4]
  reshaped <- array(x, dim = c(B, H %/% 2, 2, W %/% 2, 2, C))
  apply(reshaped, c(1, 2, 4, 6), max)
}

vgg_features <- function(x, config) {
  out <- x
  for (layer in config) {
    if (is.numeric(layer)) {
      out <- conv_relu(out, layer)
    } else if (layer == "M") {
      out <- maxpool_2x2(out)
    }
  }
  out
}
