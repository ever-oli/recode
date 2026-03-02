import mlx.core as mx
from typing import Tuple


def apply_mlm_mask(
    token_ids: mx.array,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[mx.array, mx.array, mx.array]:
    if seed is not None:
        mx.random.seed(seed)

    masked_ids = mx.array(token_ids)
    labels = mx.full(token_ids.shape, -100)

    mask_eligible = mx.logical_not(mx.isin(token_ids, mx.array([101, 102, 0])))
    probability_matrix = mx.random.uniform(shape=token_ids.shape)
    mask_indices = mx.logical_and(probability_matrix < mask_prob, mask_eligible)

    labels = mx.where(mask_indices, token_ids, labels)

    random_dispatch = mx.random.uniform(shape=token_ids.shape)
    indices_replaced = mx.logical_and(mask_indices, random_dispatch < 0.8)
    masked_ids = mx.where(indices_replaced, mask_token_id, masked_ids)

    indices_random = mx.logical_and(mask_indices, mx.logical_and(random_dispatch >= 0.8, random_dispatch < 0.9))
    random_tokens = mx.random.randint(0, vocab_size, shape=token_ids.shape)
    masked_ids = mx.where(indices_random, random_tokens, masked_ids)

    return masked_ids, labels, mask_indices


class MLMHead:
    """Masked LM prediction head."""

    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = mx.random.normal(shape=(hidden_size, vocab_size)) * 0.02
        self.b = mx.zeros((vocab_size,))

    def forward(self, hidden_states: mx.array) -> mx.array:
        return mx.matmul(hidden_states, self.W) + self.b
