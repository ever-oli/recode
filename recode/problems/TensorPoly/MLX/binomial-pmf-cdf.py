import mlx.core as mx


def binomial_pmf_cdf(n, p, k):
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")

    ks = mx.arange(0, k + 1)
    log_coeff = mx.log(mx.exp(mx.lgamma(n + 1) - mx.lgamma(ks + 1) - mx.lgamma(n - ks + 1)))
    log_pmf = log_coeff + ks * mx.log(p) + (n - ks) * mx.log(1 - p)
    pmf = mx.exp(log_pmf[-1])
    cdf = mx.sum(mx.exp(log_pmf))

    return float(pmf.item()), float(cdf.item())
