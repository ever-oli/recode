scaled_dot_product_attention <- function(Q, K, V) {
  dims <- dim(Q)
  batch_size <- dims[1]
  seq_len_q <- dims[2]
  d_k <- dims[3]
  d_v <- dim(V)[3]

  output <- array(0, dim = c(batch_size, seq_len_q, d_v))

  for (b in seq_len(batch_size)) {
    scores <- Q[b, , ] %*% t(K[b, , ])
    scaled_scores <- scores / sqrt(d_k)
    exp_scores <- exp(scaled_scores - apply(scaled_scores, 1, max))
    attention_weights <- exp_scores / rowSums(exp_scores)
    output[b, , ] <- attention_weights %*% V[b, , ]
  }

  output
}
