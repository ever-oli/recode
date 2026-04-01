binomial_pmf_cdf <- function(n, p, k) {
  if (p < 0 || p > 1) {
    stop("p must be in [0, 1]")
  }
  if (k < 0 || k > n) {
    stop("k must be in [0, n]")
  }

  pmf <- dbinom(k, size = n, prob = p)
  cdf <- pbinom(k, size = n, prob = p)

  list(pmf = as.numeric(pmf), cdf = as.numeric(cdf))
}
