function add_position_embedding(patches, num_patches::Int, embed_dim::Int)
    position_embeddings = randn(1, num_patches, embed_dim) .* 0.01
    patches .+ position_embeddings
end
