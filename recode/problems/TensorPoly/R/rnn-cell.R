rnn_cell <- function(x_t, h_prev, W_xh, W_hh, b_h) {
  input_term <- x_t %*% t(W_xh)
  hidden_term <- h_prev %*% t(W_hh)
  tanh(input_term + hidden_term + b_h)
}
