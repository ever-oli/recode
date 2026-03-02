import mlx.core as mx


def add_position_embedding(patches: mx.array, num_patches: int, embed_dim: int) -> mx.array:
    position_embeddings = mx.random.normal(shape=(1, num_patches, embed_dim)) * 0.01
    return patches + position_embeddings
