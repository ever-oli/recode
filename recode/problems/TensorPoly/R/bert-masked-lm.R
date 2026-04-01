apply_mlm_mask <- function(token_ids, vocab_size, mask_token_id = 103, mask_prob = 0.15, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  masked_ids <- token_ids
  labels <- matrix(-100, nrow = nrow(token_ids), ncol = ncol(token_ids))

  mask_eligible <- !(token_ids %in% c(101, 102, 0))
  probability_matrix <- matrix(runif(length(token_ids)), nrow = nrow(token_ids))
  mask_indices <- (probability_matrix < mask_prob) & mask_eligible

  labels[mask_indices] <- token_ids[mask_indices]

  random_dispatch <- matrix(runif(length(token_ids)), nrow = nrow(token_ids))
  indices_replaced <- mask_indices & (random_dispatch < 0.8)
  masked_ids[indices_replaced] <- mask_token_id

  indices_random <- mask_indices & (random_dispatch >= 0.8) & (random_dispatch < 0.9)
  masked_ids[indices_random] <- sample(0:(vocab_size - 1), sum(indices_random), replace = TRUE)

  list(masked_ids = masked_ids, labels = labels, mask_indices = mask_indices)
}

MLMHead <- setRefClass(
  "MLMHead",
  fields = list(
    hidden_size = "numeric",
    vocab_size = "numeric",
    W = "matrix",
    b = "numeric"
  ),
  methods = list(
    initialize = function(hidden_size, vocab_size) {
      hidden_size <<- hidden_size
      vocab_size <<- vocab_size
      W <<- matrix(rnorm(hidden_size * vocab_size, sd = 0.02), nrow = hidden_size, ncol = vocab_size)
      b <<- numeric(vocab_size)
    },
    forward = function(hidden_states) {
      hidden_states %*% W + b
    }
  )
)
