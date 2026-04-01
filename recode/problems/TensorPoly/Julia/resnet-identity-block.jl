relu(x) = max.(0, x)

mutable struct IdentityBlock
    channels::Int
    W1
    W2
end

function IdentityBlock(channels::Int)
    W1 = randn(channels, channels) .* 0.01
    W2 = randn(channels, channels) .* 0.01
    IdentityBlock(channels, W1, W2)
end

function forward(block::IdentityBlock, x)
    identity = x
    out = relu(x * block.W1)
    out = out * block.W2
    out .+ identity
end
