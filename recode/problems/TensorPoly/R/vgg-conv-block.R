vgg_conv_block <- function(x, num_convs, out_channels) {
  current_x <- x

  for (i in seq_len(num_convs)) {
    in_channels <- dim(current_x)[4]
    limit <- sqrt(2 / (3 * 3 * in_channels))
    weights <- array(rnorm(3 * 3 * in_channels * out_channels) * limit, dim = c(3, 3, in_channels, out_channels))
    bias <- numeric(out_channels)

    padded_x <- array(0, dim = c(dim(current_x)[1], dim(current_x)[2] + 2, dim(current_x)[3] + 2, in_channels))
    padded_x[, 2:(dim(current_x)[2] + 1), 2:(dim(current_x)[3] + 1), ] <- current_x

    batch <- dim(current_x)[1]
    h <- dim(current_x)[2]
    w <- dim(current_x)[3]
    out <- array(0, dim = c(batch, h, w, out_channels))

    for (i in 1:3) {
      for (j in 1:3) {
        window <- padded_x[, i:(i + h - 1), j:(j + w - 1), ]
        out <- out + apply(window, c(1, 2, 3), function(slice) slice %*% weights[i, j, , ])
      }
    }

    out <- out + bias
    current_x <- pmax(0, out)
  }

  current_x
}
