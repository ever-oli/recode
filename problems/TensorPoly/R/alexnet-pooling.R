max_pool2d <- function(x, kernel_size = 3, stride = 2) {
  dims <- dim(x)
  batch_size <- dims[1]
  h_in <- dims[2]
  w_in <- dims[3]
  channels <- dims[4]

  h_out <- (h_in - kernel_size) %/% stride + 1
  w_out <- (w_in - kernel_size) %/% stride + 1

  array(0, dim = c(batch_size, h_out, w_out, channels))
}
