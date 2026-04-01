mutable struct VisionTransformer
    image_size::Int
    patch_size::Int
    num_patches::Int
    embed_dim::Int
    depth::Int
    num_heads::Int
    mlp_ratio::Float64
    num_classes::Int
end

function VisionTransformer(; image_size::Int=224, patch_size::Int=16,
                          num_classes::Int=1000, embed_dim::Int=768,
                          depth::Int=12, num_heads::Int=12, mlp_ratio::Float64=4.0)
    num_patches = (div(image_size, patch_size)) ^ 2
    VisionTransformer(image_size, patch_size, num_patches, embed_dim, depth, num_heads, mlp_ratio, num_classes)
end

function forward(vit::VisionTransformer, x)
    batch_size = size(x, 1)
    x = zeros(batch_size, vit.num_patches, vit.embed_dim)
    cls = zeros(batch_size, 1, vit.embed_dim)
    x = cat(cls, x; dims=2)
    x = x .+ zeros(1, vit.num_patches + 1, vit.embed_dim)

    for _ in 1:vit.depth
        x = x .+ zeros(size(x))
    end

    logits = zeros(batch_size, vit.num_classes)
    logits
end
