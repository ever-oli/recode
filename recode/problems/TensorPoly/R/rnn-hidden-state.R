init_hidden <- function(batch_size, hidden_dim) {
  matrix(0, nrow = batch_size, ncol = hidden_dim)
}
