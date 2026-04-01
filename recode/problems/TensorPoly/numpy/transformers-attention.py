import math
import numpy as np


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1))
    scaled_scores = scores / math.sqrt(d_k)

    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = np.matmul(attention_weights, V)
    return output
