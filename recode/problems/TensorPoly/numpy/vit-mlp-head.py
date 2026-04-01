import numpy as np


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    """
    cls_token = encoder_output[:, 0, :]
    cls_norm = layer_norm(cls_token)

    embed_dim = cls_token.shape[-1]
    W = np.random.randn(embed_dim, num_classes) * 0.01
    b = np.zeros(num_classes)

    logits = np.matmul(cls_norm, W) + b
    return logits
