softmax(x; dims=-1) = exp.(x .- maximum(x, dims=dims)) ./ sum(exp.(x .- maximum(x, dims=dims)), dims=dims)

function multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads::Int)
    batch_size, seq_len, d_model = size(Q)
    d_k = div(d_model, num_heads)

    Q_proj = Q * W_q
    K_proj = K * W_k
    V_proj = V * W_v

    Q_heads = reshape(Q_proj, batch_size, seq_len, num_heads, d_k)
    K_heads = reshape(K_proj, batch_size, seq_len, num_heads, d_k)
    V_heads = reshape(V_proj, batch_size, seq_len, num_heads, d_k)

    Q_trans = permutedims(Q_heads, (1, 3, 2, 4))
    K_trans = permutedims(K_heads, (1, 3, 2, 4))
    V_trans = permutedims(V_heads, (1, 3, 2, 4))

    scores = Q_trans * permutedims(K_trans, (1, 2, 4, 3))
    scaled_scores = scores / sqrt(d_k)
    attention_weights = softmax(scaled_scores, dims=4)
    head_outputs = attention_weights * V_trans

    head_outputs_trans = permutedims(head_outputs, (1, 3, 2, 4))
    concatenated = reshape(head_outputs_trans, batch_size, seq_len, d_model)
    output = concatenated * W_o
    return output
end
