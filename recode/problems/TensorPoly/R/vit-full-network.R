VisionTransformer <- setRefClass(
  "VisionTransformer",
  fields = list(
    image_size = "numeric",
    patch_size = "numeric",
    num_patches = "numeric",
    embed_dim = "numeric",
    depth = "numeric",
    num_heads = "numeric",
    mlp_ratio = "numeric",
    num_classes = "numeric"
  ),
  methods = list(
    initialize = function(image_size = 224, patch_size = 16,
                          num_classes = 1000, embed_dim = 768,
                          depth = 12, num_heads = 12, mlp_ratio = 4.0) {
      image_size <<- image_size
      patch_size <<- patch_size
      num_patches <<- (image_size %/% patch_size) ^ 2
      embed_dim <<- embed_dim
      depth <<- depth
      num_heads <<- num_heads
      mlp_ratio <<- mlp_ratio
      num_classes <<- num_classes
    },
    forward = function(x) {
      batch_size <- dim(x)[1]
      x <- array(0, dim = c(batch_size, num_patches, embed_dim))
      cls <- array(0, dim = c(batch_size, 1, embed_dim))
      x <- array(c(cls, x), dim = c(batch_size, num_patches + 1, embed_dim))
      x <- x + array(0, dim = c(1, num_patches + 1, embed_dim))

      for (i in seq_len(depth)) {
        x <- x + array(0, dim = dim(x))
      }

      logits <- array(0, dim = c(batch_size, num_classes))
      logits
    }
  )
)
