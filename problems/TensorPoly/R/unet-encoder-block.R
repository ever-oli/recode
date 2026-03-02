unet_encoder_block <- function(x, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]

  skip_H <- H - 4
  skip_W <- W - 4
  skip_out <- array(0, dim = c(batch, skip_H, skip_W, out_channels))

  pool_H <- skip_H %/% 2
  pool_W <- skip_W %/% 2
  pool_out <- array(0, dim = c(batch, pool_H, pool_W, out_channels))

  list(pool_out = pool_out, skip_out = skip_out)
}
