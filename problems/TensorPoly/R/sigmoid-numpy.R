sigmoid <- function(x) {
  x_arr <- as.numeric(x)
  1.0 / (1.0 + exp(-x_arr))
}
