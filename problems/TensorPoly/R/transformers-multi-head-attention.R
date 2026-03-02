softmax <- function(x, axis = -1) {
  exp_x <- exp(x - apply(x, axis, max))
  exp_x / apply(exp_x, axis, sum)
}

multi_head_attention <- function(Q, K, V, W_q, W_k, W_v, W_o, num_heads) {
  dims <- dim(Q)
  batch_size <- dims[1]
  seq_len <- dims[2]
  d_model <- dims[3]
  d_k <- d_model %/% num_heads

  Q_proj <- array(0, dim = c(batch_size, seq_len, d_model))
  K_proj <- array(0, dim = c(batch_size, seq_len, d_model))
  V_proj <- array(0, dim = c(batch_size, seq_len, d_model))

  for (b in seq_len(batch_size)) {
    Q_proj[b, , ] <- Q[b, , ] %*% W_q
    K_proj[b, , ] <- K[b, , ] %*% W_k
    V_proj[b, , ] <- V[b, , ] %*% W_v
  }

  head_outputs <- array(0, dim = c(batch_size, num_heads, seq_len, d_k))

  for (b in seq_len(batch_size)) {
    for (h in seq_len(num_heads)) {
      idx <- ((h - 1) * d_k + 1):(h * d_k)
      Qh <- Q_proj[b, , idx]
      Kh <- K_proj[b, , idx]
      Vh <- V_proj[b, , idx]

      scores <- Qh %*% t(Kh)
      scaled_scores <- scores / sqrt(d_k)
      exp_scores <- exp(scaled_scores - apply(scaled_scores, 1, max))
      attention_weights <- exp_scores / rowSums(exp_scores)
      head_outputs[b, h, , ] <- attention_weights %*% Vh
    }
  }

  concatenated <- array(0, dim = c(batch_size, seq_len, d_model))
  for (b in seq_len(batch_size)) {
    concat_rows <- list()
    for (h in seq_len(num_heads)) {
      concat_rows[[h]] <- head_outputs[b, h, , ]
    }
    concatenated[b, , ] <- do.call(cbind, concat_rows)
  }

  output <- array(0, dim = c(batch_size, seq_len, d_model))
  for (b in seq_len(batch_size)) {
    output[b, , ] <- concatenated[b, , ] %*% W_o
  }

  output
}
