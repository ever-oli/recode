import mlx.core as mx


def patch_embed(image: mx.array, patch_size: int, embed_dim: int) -> mx.array:
    batch, H, W, C = image.shape

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = mx.reshape(
        image,
        (batch, num_patches_h, patch_size, num_patches_w, patch_size, C)
    )

    patches = mx.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches_flat = mx.reshape(
        patches,
        (batch, num_patches_h, num_patches_w, patch_size * patch_size * C)
    )
    patches_seq = mx.reshape(patches_flat, (batch, num_patches, patch_size * patch_size * C))

    patch_dim = patch_size * patch_size * C
    W_proj = mx.random.normal(shape=(patch_dim, embed_dim)) * 0.01
    embeddings = mx.matmul(patches_seq, W_proj)
    return embeddings
