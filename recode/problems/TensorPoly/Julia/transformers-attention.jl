function scaled_dot_product_attention(Q, K, V)
    d_k = size(Q, ndims(Q))
    scores = Q * permutedims(K, (1, 3, 2))
    scaled_scores = scores / sqrt(d_k)

    exp_scores = exp.(scaled_scores .- maximum(scaled_scores, dims=3))
    attention_weights = exp_scores ./ sum(exp_scores, dims=3)

    output = attention_weights * V
    return output
end
