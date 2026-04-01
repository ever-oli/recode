feed_forward <- function(x, W1, b1, W2, b2) {
  hidden <- x %*% W1 + b1
  relu_out <- pmax(0, hidden)
  relu_out %*% W2 + b2
}
