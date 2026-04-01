import torch


def add_position_embedding(patches: torch.Tensor, num_patches: int, embed_dim: int) -> torch.Tensor:
    position_embeddings = torch.randn(1, num_patches, embed_dim) * 0.01
    return patches + position_embeddings
