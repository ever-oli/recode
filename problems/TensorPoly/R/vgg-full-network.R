vgg16 <- function(x, num_classes = 1000) {
  vgg16_config <- list(
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M"
  )

  features <- vgg_features(x, vgg16_config)
  vgg_classifier(features, num_classes)
}
