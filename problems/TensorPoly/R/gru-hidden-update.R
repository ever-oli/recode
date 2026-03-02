hidden_update <- function(h_prev, h_tilde, z_t) {
  keep_old <- z_t * h_prev
  use_new <- (1 - z_t) * h_tilde
  keep_old + use_new
}
