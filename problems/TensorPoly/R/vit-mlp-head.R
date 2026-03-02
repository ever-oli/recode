layer_norm <- function(x, eps = 1e-6) {
  mean <- apply(x, length(dim(x)), mean)
  var <- apply(x, length(dim(x)), var)
  x_normalized <- (x - mean) / sqrt(var + eps)
  x_normalized
}

classification_head <- function(encoder_output, num_classes) {
  cls_token <- encoder_output[, 1, ]
  cls_norm <- layer_norm(cls_token)

  embed_dim <- dim(cls_norm)[2]
  W <- matrix(rnorm(embed_dim * num_classes, sd = 0.01), nrow = embed_dim, ncol = num_classes)
  b <- numeric(num_classes)

  cls_norm %*% W + b
}
