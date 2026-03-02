softmax(x; dims=-1) = exp.(x .- maximum(x, dims=dims)) ./ sum(exp.(x .- maximum(x, dims=dims)), dims=dims)

function layer_norm(x, gamma, beta; eps=1e-6)
    mean_vals = mean(x, dims=ndims(x))
    var_vals = var(x, dims=ndims(x))
    x_normalized = (x .- mean_vals) ./ sqrt.(var_vals .+ eps)
    gamma .* x_normalized .+ beta
end

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

function feed_forward(x, W1, b1, W2, b2)
    hidden = x * W1 .+ b1
    relu_out = max.(0, hidden)
    relu_out * W2 .+ b2
end

function encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                       gamma1, beta1, gamma2, beta2, num_heads::Int)
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_attn_residual = x .+ attn_output
    x_norm1 = layer_norm(x_attn_residual, gamma1, beta1)

    ff_output = feed_forward(x_norm1, W1, b1, W2, b2)
    x_ff_residual = x_norm1 .+ ff_output
    layer_norm(x_ff_residual, gamma2, beta2)
end
