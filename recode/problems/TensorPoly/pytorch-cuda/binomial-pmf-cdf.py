import math
import torch


def binomial_pmf_cdf(n, p, k, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _ = torch.tensor(0.0, device=device)

    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")

    pmf = math.comb(int(n), int(k)) * (p ** k) * ((1 - p) ** (n - k))
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(int(n), int(i)) * (p ** i) * ((1 - p) ** (n - i))

    return float(pmf), float(cdf)
