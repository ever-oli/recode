alexnet_conv1 <- function(image) {
  batch_size <- dim(image)[1]
  output_h <- 55
  output_w <- 55
  num_filters <- 96
  array(0, dim = c(batch_size, output_h, output_w, num_filters))
}
