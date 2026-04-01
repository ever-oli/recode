function prepend_class_token(patches, embed_dim::Int)
    batch_size = size(patches, 1)
    cls_token = randn(1, 1, embed_dim) .* 0.02
    cls_token_batch = repeat(cls_token, batch_size, 1, 1)
    cat(cls_token_batch, patches; dims=2)
end
