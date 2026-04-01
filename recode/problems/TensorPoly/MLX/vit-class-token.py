import mlx.core as mx


def prepend_class_token(patches: mx.array, embed_dim: int) -> mx.array:
    batch_size = patches.shape[0]
    cls_token = mx.random.normal(shape=(1, 1, embed_dim)) * 0.02
    cls_token_batch = mx.repeat(cls_token, repeats=batch_size, axis=0)
    return mx.concatenate([cls_token_batch, patches], axis=1)
