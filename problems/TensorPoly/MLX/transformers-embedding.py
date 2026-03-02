import math
import mlx.core as mx


def create_embedding_layer(vocab_size: int, d_model: int) -> mx.array:
    return mx.random.normal(shape=(vocab_size, d_model)) * (1.0 / math.sqrt(d_model))


def embed_tokens(embedding: mx.array, tokens: mx.array, d_model: int) -> mx.array:
    embedded = embedding[tokens]
    return embedded * math.sqrt(d_model)
