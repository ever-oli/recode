random_crop <- function(image, crop_size = 224) {
  dims <- dim(image)
  h <- dims[1]
  w <- dims[2]

  top <- sample.int(h - crop_size + 1, 1)
  left <- sample.int(w - crop_size + 1, 1)

  image[top:(top + crop_size - 1), left:(left + crop_size - 1), ]
}


random_horizontal_flip <- function(image, p = 0.5) {
  if (runif(1) < p) {
    return(image[, ncol(image):1, ])
  }
  image
}
