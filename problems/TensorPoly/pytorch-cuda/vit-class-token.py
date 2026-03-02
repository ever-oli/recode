import torch


def prepend_class_token(patches: torch.Tensor, embed_dim: int, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = patches.size(0)
    cls_token = torch.randn(1, 1, embed_dim, device=device) * 0.02
    cls_token_batch = cls_token.repeat(batch_size, 1, 1)
    return torch.cat([cls_token_batch, patches.to(device)], dim=1)
