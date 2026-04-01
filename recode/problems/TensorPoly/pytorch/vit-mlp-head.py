import torch


def layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def classification_head(encoder_output: torch.Tensor, num_classes: int) -> torch.Tensor:
    cls_token = encoder_output[:, 0, :]
    cls_norm = layer_norm(cls_token)

    embed_dim = cls_token.shape[-1]
    W = torch.randn(embed_dim, num_classes) * 0.01
    b = torch.zeros(num_classes)

    logits = torch.matmul(cls_norm, W) + b
    return logits
