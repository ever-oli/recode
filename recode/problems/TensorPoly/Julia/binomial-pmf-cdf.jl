function binomial_pmf_cdf(n, p, k)
    if p < 0 || p > 1
        error("p must be in [0, 1]")
    end
    if k < 0 || k > n
        error("k must be in [0, n]")
    end

    pmf = binomial(n, k) * (p ^ k) * ((1 - p) ^ (n - k))
    cdf = 0.0
    for i in 0:k
        cdf += binomial(n, i) * (p ^ i) * ((1 - p) ^ (n - i))
    end

    return (pmf = float(pmf), cdf = float(cdf))
end
