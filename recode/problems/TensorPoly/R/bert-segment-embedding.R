BertEmbeddings <- setRefClass(
  "BertEmbeddings",
  fields = list(
    hidden_size = "numeric",
    token_embeddings = "matrix",
    position_embeddings = "matrix",
    segment_embeddings = "matrix"
  ),
  methods = list(
    initialize = function(vocab_size, max_position, hidden_size) {
      hidden_size <<- hidden_size
      token_embeddings <<- matrix(rnorm(vocab_size * hidden_size, sd = 0.02), nrow = vocab_size, ncol = hidden_size)
      position_embeddings <<- matrix(rnorm(max_position * hidden_size, sd = 0.02), nrow = max_position, ncol = hidden_size)
      segment_embeddings <<- matrix(rnorm(2 * hidden_size, sd = 0.02), nrow = 2, ncol = hidden_size)
    },
    forward = function(token_ids, segment_ids) {
      tok_emb <- token_embeddings[token_ids + 1, , drop = FALSE]
      seq_len <- ncol(token_ids)
      positions <- 1:seq_len
      pos_emb <- position_embeddings[positions, , drop = FALSE]
      seg_emb <- segment_embeddings[segment_ids + 1, , drop = FALSE]
      tok_emb + pos_emb + seg_emb
    }
  )
)
