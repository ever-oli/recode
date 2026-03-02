import torch


def add_position_embedding(patches: torch.Tensor, num_patches: int, embed_dim: int, device=None) -> torch.Tensor:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    position_embeddings = torch.randn(1, num_patches, embed_dim, device=device) * 0.01
    return patches.to(device) + position_embeddings
