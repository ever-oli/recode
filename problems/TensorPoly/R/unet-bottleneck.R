unet_bottleneck <- function(x, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]

  H_out <- H - 4
  W_out <- W - 4
  array(0, dim = c(batch, H_out, W_out, out_channels))
}
