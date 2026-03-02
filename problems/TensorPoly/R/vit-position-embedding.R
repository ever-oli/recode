add_position_embedding <- function(patches, num_patches, embed_dim) {
  position_embeddings <- array(rnorm(num_patches * embed_dim, sd = 0.01), dim = c(1, num_patches, embed_dim))
  patches + position_embeddings
}
