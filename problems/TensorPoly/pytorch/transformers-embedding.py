import math
import torch
import torch.nn as nn


def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    embedding = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    embedded = embedding(tokens)
    return embedded * math.sqrt(d_model)
