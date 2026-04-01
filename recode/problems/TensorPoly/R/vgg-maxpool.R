vgg_maxpool <- function(x) {
  batch <- dim(x)[1]
  h <- dim(x)[2]
  w <- dim(x)[3]
  c <- dim(x)[4]

  reshaped_x <- array(x, dim = c(batch, h %/% 2, 2, w %/% 2, 2, c))
  apply(reshaped_x, c(1, 2, 4, 6), max)
}
