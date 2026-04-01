prepend_class_token <- function(patches, embed_dim) {
  batch_size <- dim(patches)[1]
  cls_token <- array(rnorm(embed_dim, sd = 0.02), dim = c(1, 1, embed_dim))
  cls_token_batch <- array(rep(cls_token, batch_size), dim = c(batch_size, 1, embed_dim))
  array(c(cls_token_batch, patches), dim = c(batch_size, dim(patches)[2] + 1, embed_dim))
}
