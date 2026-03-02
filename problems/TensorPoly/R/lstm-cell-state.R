update_cell_state <- function(C_prev, f_t, i_t, c_tilde) {
  f_t * C_prev + i_t * c_tilde
}
