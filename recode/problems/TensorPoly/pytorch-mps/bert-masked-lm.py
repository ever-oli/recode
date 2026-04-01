import torch
from typing import Tuple


def apply_mlm_mask(
    token_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)

    masked_ids = token_ids.clone()
    labels = torch.full(token_ids.shape, -100, device=token_ids.device)

    mask_eligible = ~torch.isin(token_ids, torch.tensor([101, 102, 0], device=token_ids.device))
    probability_matrix = torch.rand_like(token_ids.float())
    mask_indices = (probability_matrix < mask_prob) & mask_eligible

    labels[mask_indices] = token_ids[mask_indices]

    random_dispatch = torch.rand_like(token_ids.float())
    indices_replaced = mask_indices & (random_dispatch < 0.8)
    masked_ids[indices_replaced] = mask_token_id

    indices_random = mask_indices & (random_dispatch >= 0.8) & (random_dispatch < 0.9)
    masked_ids[indices_random] = torch.randint(0, vocab_size, size=(indices_random.sum(),), device=token_ids.device)

    return masked_ids, labels, mask_indices


class MLMHead:
    """Masked LM prediction head."""

    def __init__(self, hidden_size: int, vocab_size: int, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = torch.randn(hidden_size, vocab_size, device=device) * 0.02
        self.b = torch.zeros(vocab_size, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.W) + self.b
