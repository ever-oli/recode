create_embedding_layer <- function(vocab_size, d_model) {
  matrix(rnorm(vocab_size * d_model, sd = 1 / sqrt(d_model)), nrow = vocab_size, ncol = d_model)
}

embed_tokens <- function(embedding, tokens, d_model) {
  embedded <- embedding[tokens + 1, , drop = FALSE]
  scaled_embeddings <- embedded * sqrt(d_model)
  scaled_embeddings
}
