import math
import numpy as np


def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial(n, p) PMF at k and CDF at k.
    Returns (pmf, cdf) as scalar floats.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if not (0 <= k <= n):
        raise ValueError("k must be in [0, n]")

    C_nk = math.comb(int(n), int(k))
    pmf = C_nk * (p ** k) * ((1 - p) ** (n - k))

    cdf = 0.0
    for i in range(0, k + 1):
        C_ni = math.comb(int(n), int(i))
        cdf += C_ni * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)
