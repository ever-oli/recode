function create_embedding_layer(vocab_size::Int, d_model::Int)
    randn(vocab_size, d_model) .* (1 / sqrt(d_model))
end

function embed_tokens(embedding, tokens, d_model::Int)
    embedded = embedding[tokens .+ 1, :]
    embedded .* sqrt(d_model)
end
