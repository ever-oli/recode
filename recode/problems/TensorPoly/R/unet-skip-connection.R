crop_and_concat <- function(encoder_features, decoder_features) {
  dims_enc <- dim(encoder_features)
  dims_dec <- dim(decoder_features)

  H_enc <- dims_enc[2]
  W_enc <- dims_enc[3]
  H_dec <- dims_dec[2]
  W_dec <- dims_dec[3]

  crop_h <- (H_enc - H_dec) %/% 2
  crop_w <- (W_enc - W_dec) %/% 2

  encoder_cropped <- encoder_features[, (crop_h + 1):(crop_h + H_dec), (crop_w + 1):(crop_w + W_dec), ]
  array(c(encoder_cropped, decoder_features), dim = c(dims_dec[1], H_dec, W_dec, dims_enc[4] + dims_dec[4]))
}
