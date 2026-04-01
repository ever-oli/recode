import numpy as np


def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    batch_size = patches.shape[0]
    cls_token = np.random.randn(1, 1, embed_dim) * 0.02
    cls_token_batch = np.repeat(cls_token, batch_size, axis=0)
    return np.concatenate([cls_token_batch, patches], axis=1)
