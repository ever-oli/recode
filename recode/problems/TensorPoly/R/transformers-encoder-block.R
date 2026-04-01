softmax <- function(x, axis = -1) {
  exp_x <- exp(x - apply(x, axis, max))
  exp_x / apply(exp_x, axis, sum)
}

layer_norm <- function(x, gamma, beta, eps = 1e-6) {
  dims <- dim(x)
  keep_axes <- seq_len(length(dims) - 1)
  mean_vals <- apply(x, keep_axes, mean)
  var_vals <- apply(x, keep_axes, var)
  mean_arr <- array(mean_vals, dim = c(dims[-length(dims)], 1))
  var_arr <- array(var_vals, dim = c(dims[-length(dims)], 1))
  x_normalized <- (x - mean_arr) / sqrt(var_arr + eps)
  gamma * x_normalized + beta
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

feed_forward <- function(x, W1, b1, W2, b2) {
  hidden <- x %*% W1 + b1
  relu_out <- pmax(0, hidden)
  relu_out %*% W2 + b2
}

encoder_block <- function(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2, gamma1, beta1, gamma2, beta2, num_heads) {
  attn_output <- multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
  x_attn_residual <- x + attn_output
  x_norm1 <- layer_norm(x_attn_residual, gamma1, beta1)

  ff_output <- feed_forward(x_norm1, W1, b1, W2, b2)
  x_ff_residual <- x_norm1 + ff_output
  layer_norm(x_ff_residual, gamma2, beta2)
}
