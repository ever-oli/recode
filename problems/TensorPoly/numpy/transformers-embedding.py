import math
import numpy as np


def create_embedding_layer(vocab_size: int, d_model: int) -> np.ndarray:
    """
    Create an embedding layer.
    """
    embedding = np.random.randn(vocab_size, d_model) * (1.0 / math.sqrt(d_model))
    return embedding


def embed_tokens(embedding: np.ndarray, tokens: np.ndarray, d_model: int) -> np.ndarray:
    """
    Convert token indices to scaled embeddings.
    """
    embedded = embedding[tokens]
    scaled_embeddings = embedded * math.sqrt(d_model)
    return scaled_embeddings
