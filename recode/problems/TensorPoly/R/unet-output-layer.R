unet_output <- function(features, num_classes) {
  dims <- dim(features)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]
  array(0, dim = c(batch, H, W, num_classes))
}
