function layer_norm(x; eps=1e-6)
    mean_vals = mean(x, dims=ndims(x))
    var_vals = var(x, dims=ndims(x))
    (x .- mean_vals) ./ sqrt.(var_vals .+ eps)
end

function gelu(x)
    0.5 .* x .* (1 .+ tanh.(sqrt(2 / pi) .* (x .+ 0.044715 .* x .^ 3)))
end

softmax(x; dims=-1) = exp.(x .- maximum(x, dims=dims)) ./ sum(exp.(x .- maximum(x, dims=dims)), dims=dims)

function multi_head_self_attention(x, num_heads::Int, embed_dim::Int)
    batch, seq_len, _ = size(x)
    head_dim = div(embed_dim, num_heads)

    W_q = randn(embed_dim, embed_dim) .* 0.02
    W_k = randn(embed_dim, embed_dim) .* 0.02
    W_v = randn(embed_dim, embed_dim) .* 0.02
    W_o = randn(embed_dim, embed_dim) .* 0.02

    Q = x * W_q
    K = x * W_k
    V = x * W_v

    Q = reshape(Q, batch, seq_len, num_heads, head_dim)
    K = reshape(K, batch, seq_len, num_heads, head_dim)
    V = reshape(V, batch, seq_len, num_heads, head_dim)

    Q = permutedims(Q, (1, 3, 2, 4))
    K = permutedims(K, (1, 3, 2, 4))
    V = permutedims(V, (1, 3, 2, 4))

    scores = Q * permutedims(K, (1, 2, 4, 3)) / sqrt(head_dim)
    attn_weights = softmax(scores, dims=4)
    attn_output = attn_weights * V

    attn_output = permutedims(attn_output, (1, 3, 2, 4))
    attn_output = reshape(attn_output, batch, seq_len, embed_dim)
    attn_output * W_o
end

function mlp(x, embed_dim::Int, mlp_ratio::Float64)
    hidden_dim = Int(embed_dim * mlp_ratio)
    W1 = randn(embed_dim, hidden_dim) .* 0.02
    b1 = zeros(hidden_dim)
    W2 = randn(hidden_dim, embed_dim) .* 0.02
    b2 = zeros(embed_dim)
    h = gelu(x * W1 .+ b1)
    h * W2 .+ b2
end

function vit_encoder_block(x, embed_dim::Int, num_heads::Int; mlp_ratio::Float64=4.0)
    x_norm1 = layer_norm(x)
    attn_output = multi_head_self_attention(x_norm1, num_heads, embed_dim)
    x = x .+ attn_output

    x_norm2 = layer_norm(x)
    mlp_output = mlp(x_norm2, embed_dim, mlp_ratio)
    x .+ mlp_output
end
