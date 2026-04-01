encoder_block <- function(x, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]

  skip_H <- H - 4
  skip_W <- W - 4
  skip <- array(0, dim = c(batch, skip_H, skip_W, out_channels))

  pool_H <- skip_H %/% 2
  pool_W <- skip_W %/% 2
  pooled <- array(0, dim = c(batch, pool_H, pool_W, out_channels))

  list(pooled = pooled, skip = skip)
}

bottleneck <- function(x, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]
  array(0, dim = c(batch, H - 4, W - 4, out_channels))
}

decoder_block <- function(x, skip, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]

  H_up <- H * 2
  W_up <- W * 2

  skip_dims <- dim(skip)
  H_skip <- skip_dims[2]
  W_skip <- skip_dims[3]
  crop_h <- (H_skip - H_up) %/% 2
  crop_w <- (W_skip - W_up) %/% 2
  _ <- skip[, (crop_h + 1):(crop_h + H_up), (crop_w + 1):(crop_w + W_up), ]

  H_out <- H_up - 4
  W_out <- W_up - 4
  array(0, dim = c(batch, H_out, W_out, out_channels))
}

output_layer <- function(x, num_classes) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]
  array(0, dim = c(batch, H, W, num_classes))
}

unet <- function(x, num_classes = 2) {
  e1 <- encoder_block(x, out_channels = 64)
  e2 <- encoder_block(e1$pooled, out_channels = 128)
  e3 <- encoder_block(e2$pooled, out_channels = 256)
  e4 <- encoder_block(e3$pooled, out_channels = 512)

  bottleneck_out <- bottleneck(e4$pooled, out_channels = 1024)

  d4_out <- decoder_block(bottleneck_out, e4$skip, out_channels = 512)
  d3_out <- decoder_block(d4_out, e3$skip, out_channels = 256)
  d2_out <- decoder_block(d3_out, e2$skip, out_channels = 128)
  d1_out <- decoder_block(d2_out, e1$skip, out_channels = 64)

  output_layer(d1_out, num_classes)
}
