unet_decoder_block <- function(x, skip, out_channels) {
  dims <- dim(x)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]

  skip_dims <- dim(skip)
  H_skip <- skip_dims[2]
  W_skip <- skip_dims[3]

  H_up <- H * 2
  W_up <- W * 2

  crop_h <- (H_skip - H_up) %/% 2
  crop_w <- (W_skip - W_up) %/% 2
  _ <- skip[, (crop_h + 1):(crop_h + H_up), (crop_w + 1):(crop_w + W_up), ]

  H_out <- H_up - 4
  W_out <- W_up - 4

  array(0, dim = c(batch, H_out, W_out, out_channels))
}
