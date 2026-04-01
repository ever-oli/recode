import torch


def patch_embed(image: torch.Tensor, patch_size: int, embed_dim: int) -> torch.Tensor:
    batch, H, W, C = image.shape

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    patches = image.reshape(
        batch,
        num_patches_h, patch_size,
        num_patches_w, patch_size,
        C
    )

    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches_flat = patches.reshape(batch, num_patches_h, num_patches_w, patch_size * patch_size * C)
    patches_seq = patches_flat.reshape(batch, num_patches, patch_size * patch_size * C)

    patch_dim = patch_size * patch_size * C
    W_proj = torch.randn(patch_dim, embed_dim) * 0.01
    embeddings = torch.matmul(patches_seq, W_proj)
    return embeddings
