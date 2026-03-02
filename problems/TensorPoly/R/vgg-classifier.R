vgg_classifier <- function(features, num_classes = 1000) {
  batch_size <- dim(features)[1]
  x <- matrix(features, nrow = batch_size)

  dense_relu <- function(input_data, out_dim) {
    in_dim <- ncol(input_data)
    limit <- sqrt(2 / in_dim)
    w <- matrix(rnorm(in_dim * out_dim) * limit, nrow = in_dim, ncol = out_dim)
    b <- numeric(out_dim)
    pmax(0, input_data %*% w + b)
  }

  x <- dense_relu(x, 4096)
  x <- dense_relu(x, 4096)

  in_dim_final <- ncol(x)
  w_final <- matrix(rnorm(in_dim_final * num_classes) * sqrt(2 / in_dim_final), nrow = in_dim_final, ncol = num_classes)
  b_final <- numeric(num_classes)
  x %*% w_final + b_final
}
