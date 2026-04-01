function patch_embed(image, patch_size::Int, embed_dim::Int)
    batch, H, W, C = size(image)
    num_patches_h = div(H, patch_size)
    num_patches_w = div(W, patch_size)
    num_patches = num_patches_h * num_patches_w

    patches = reshape(image, batch,
                      num_patches_h, patch_size,
                      num_patches_w, patch_size,
                      C)
    patches = permutedims(patches, (1, 2, 4, 3, 5, 6))
    patches_flat = reshape(patches, batch, num_patches_h, num_patches_w, patch_size * patch_size * C)
    patches_seq = reshape(patches_flat, batch, num_patches, patch_size * patch_size * C)

    patch_dim = patch_size * patch_size * C
    W_proj = randn(patch_dim, embed_dim) .* 0.01

    embeddings = Array{Float64}(undef, batch, num_patches, embed_dim)
    for b in 1:batch
        embeddings[b, :, :] = patches_seq[b, :, :] * W_proj
    end
    embeddings
end
