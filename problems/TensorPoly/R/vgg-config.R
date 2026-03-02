make_vgg_config <- function(variant) {
  configs <- list(
    vgg11 = list(64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    vgg13 = list(64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    vgg16 = list(64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"),
    vgg19 = list(64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M")
  )
  key <- tolower(variant)
  if (!is.null(configs[[key]])) configs[[key]] else list()
}
