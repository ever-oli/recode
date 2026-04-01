compute_gradient_with_skip <- function(gradients_F, x) {
  grad <- x
  for (F_grad in rev(gradients_F)) {
    F_mat <- F_grad
    dim <- ncol(F_mat)
    grad <- grad %*% (diag(dim) + F_mat)
  }
  grad
}

compute_gradient_without_skip <- function(gradients_F, x) {
  grad <- x
  for (F_grad in rev(gradients_F)) {
    F_mat <- F_grad
    grad <- grad %*% F_mat
  }
  grad
}
