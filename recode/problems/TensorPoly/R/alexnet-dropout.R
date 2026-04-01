dropout <- function(x, p = 0.5, training = TRUE) {
  if (!training || p == 0) {
    return(x)
  }

  mask <- rbinom(length(x), size = 1, prob = 1 - p)
  mask <- array(mask, dim = dim(x))
  (x * mask) / (1 - p)
}
