function layer_norm(x; eps=1e-6)
    mean_vals = mean(x, dims=ndims(x))
    var_vals = var(x, dims=ndims(x))
    (x .- mean_vals) ./ sqrt.(var_vals .+ eps)
end

function classification_head(encoder_output, num_classes::Int)
    cls_token = encoder_output[:, 1, :]
    cls_norm = layer_norm(cls_token)
    embed_dim = size(cls_norm, 2)
    W = randn(embed_dim, num_classes) .* 0.01
    b = zeros(num_classes)
    cls_norm * W .+ b
end
