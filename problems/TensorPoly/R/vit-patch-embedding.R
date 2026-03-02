patch_embed <- function(image, patch_size, embed_dim) {
  dims <- dim(image)
  batch <- dims[1]
  H <- dims[2]
  W <- dims[3]
  C <- dims[4]

  num_patches_h <- H %/% patch_size
  num_patches_w <- W %/% patch_size
  num_patches <- num_patches_h * num_patches_w

  patches <- array(image, dim = c(batch, num_patches_h, patch_size, num_patches_w, patch_size, C))
  patches <- aperm(patches, c(1, 2, 4, 3, 5, 6))
  patches_flat <- array(patches, dim = c(batch, num_patches_h, num_patches_w, patch_size * patch_size * C))
  patches_seq <- array(patches_flat, dim = c(batch, num_patches, patch_size * patch_size * C))

  patch_dim <- patch_size * patch_size * C
  W_proj <- matrix(rnorm(patch_dim * embed_dim, sd = 0.01), nrow = patch_dim, ncol = embed_dim)

  embeddings <- array(0, dim = c(batch, num_patches, embed_dim))
  for (b in seq_len(batch)) {
    embeddings[b, , ] <- patches_seq[b, , ] %*% W_proj
  }

  embeddings
}
