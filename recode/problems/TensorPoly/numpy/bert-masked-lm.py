import numpy as np
from typing import Tuple


def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    masked_ids = token_ids.copy()
    labels = np.full(token_ids.shape, -100)

    mask_eligible = ~np.isin(token_ids, [101, 102, 0])
    probability_matrix = np.random.rand(*token_ids.shape)
    mask_indices = (probability_matrix < mask_prob) & mask_eligible

    labels[mask_indices] = token_ids[mask_indices]

    random_dispatch = np.random.rand(*token_ids.shape)
    indices_replaced = mask_indices & (random_dispatch < 0.8)
    masked_ids[indices_replaced] = mask_token_id

    indices_random = mask_indices & (random_dispatch >= 0.8) & (random_dispatch < 0.9)
    masked_ids[indices_random] = np.random.randint(0, vocab_size, size=np.sum(indices_random))

    return masked_ids, labels, mask_indices


class MLMHead:
    """Masked LM prediction head."""

    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        return np.dot(hidden_states, self.W) + self.b
