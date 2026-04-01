layer_norm <- function(x, eps = 1e-6) {
  mean <- apply(x, length(dim(x)), mean)
  var <- apply(x, length(dim(x)), var)
  x_normalized <- (x - mean) / sqrt(var + eps)
  x_normalized
}

gelu <- function(x) {
  0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
}

softmax <- function(x, axis = -1) {
  exp_x <- exp(x - apply(x, axis, max))
  exp_x / apply(exp_x, axis, sum)
}

multi_head_self_attention <- function(x, num_heads, embed_dim) {
  dims <- dim(x)
  batch <- dims[1]
  seq_len <- dims[2]
  head_dim <- embed_dim %/% num_heads

  W_q <- matrix(rnorm(embed_dim * embed_dim, sd = 0.02), nrow = embed_dim, ncol = embed_dim)
  W_k <- matrix(rnorm(embed_dim * embed_dim, sd = 0.02), nrow = embed_dim, ncol = embed_dim)
  W_v <- matrix(rnorm(embed_dim * embed_dim, sd = 0.02), nrow = embed_dim, ncol = embed_dim)
  W_o <- matrix(rnorm(embed_dim * embed_dim, sd = 0.02), nrow = embed_dim, ncol = embed_dim)

  Q <- array(0, dim = c(batch, seq_len, embed_dim))
  K <- array(0, dim = c(batch, seq_len, embed_dim))
  V <- array(0, dim = c(batch, seq_len, embed_dim))
  for (b in seq_len(batch)) {
    Q[b, , ] <- x[b, , ] %*% W_q
    K[b, , ] <- x[b, , ] %*% W_k
    V[b, , ] <- x[b, , ] %*% W_v
  }

  Q <- array(Q, dim = c(batch, seq_len, num_heads, head_dim))
  K <- array(K, dim = c(batch, seq_len, num_heads, head_dim))
  V <- array(V, dim = c(batch, seq_len, num_heads, head_dim))

  Q <- aperm(Q, c(1, 3, 2, 4))
  K <- aperm(K, c(1, 3, 2, 4))
  V <- aperm(V, c(1, 3, 2, 4))

  head_outputs <- array(0, dim = c(batch, num_heads, seq_len, head_dim))
  for (b in seq_len(batch)) {
    for (h in seq_len(num_heads)) {
      Qh <- Q[b, h, , ]
      Kh <- K[b, h, , ]
      Vh <- V[b, h, , ]
      scores <- Qh %*% t(Kh) / sqrt(head_dim)
      attn_weights <- softmax(scores, axis = 2)
      head_outputs[b, h, , ] <- attn_weights %*% Vh
    }
  }

  head_outputs <- aperm(head_outputs, c(1, 3, 2, 4))
  concatenated <- array(head_outputs, dim = c(batch, seq_len, embed_dim))

  output <- array(0, dim = c(batch, seq_len, embed_dim))
  for (b in seq_len(batch)) {
    output[b, , ] <- concatenated[b, , ] %*% W_o
  }

  output
}

mlp <- function(x, embed_dim, mlp_ratio) {
  hidden_dim <- as.integer(embed_dim * mlp_ratio)
  W1 <- matrix(rnorm(embed_dim * hidden_dim, sd = 0.02), nrow = embed_dim, ncol = hidden_dim)
  b1 <- numeric(hidden_dim)
  W2 <- matrix(rnorm(hidden_dim * embed_dim, sd = 0.02), nrow = hidden_dim, ncol = embed_dim)
  b2 <- numeric(embed_dim)

  h <- gelu(x %*% W1 + b1)
  h %*% W2 + b2
}

vit_encoder_block <- function(x, embed_dim, num_heads, mlp_ratio = 4.0) {
  x_norm1 <- layer_norm(x)
  attn_output <- multi_head_self_attention(x_norm1, num_heads, embed_dim)
  x <- x + attn_output

  x_norm2 <- layer_norm(x)
  mlp_output <- mlp(x_norm2, embed_dim, mlp_ratio)
  x <- x + mlp_output
  x
}
