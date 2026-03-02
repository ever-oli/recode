import torch


def prepend_class_token(patches: torch.Tensor, embed_dim: int) -> torch.Tensor:
    batch_size = patches.size(0)
    cls_token = torch.randn(1, 1, embed_dim) * 0.02
    cls_token_batch = cls_token.repeat(batch_size, 1, 1)
    return torch.cat([cls_token_batch, patches], dim=1)
