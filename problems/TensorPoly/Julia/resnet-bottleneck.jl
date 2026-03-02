relu(x) = max.(0, x)

mutable struct BottleneckBlock
    in_ch::Int
    bn_ch::Int
    out_ch::Int
    W1
    W2
    W3
    Ws
end

function BottleneckBlock(in_channels::Int, bottleneck_channels::Int, out_channels::Int)
    W1 = randn(in_channels, bottleneck_channels) .* 0.01
    W2 = randn(bottleneck_channels, bottleneck_channels) .* 0.01
    W3 = randn(bottleneck_channels, out_channels) .* 0.01
    Ws = in_channels != out_channels ? randn(in_channels, out_channels) .* 0.01 : nothing
    BottleneckBlock(in_channels, bottleneck_channels, out_channels, W1, W2, W3, Ws)
end

function forward(block::BottleneckBlock, x)
    identity = x
    out = relu(x * block.W1)
    out = relu(out * block.W2)
    out = out * block.W3
    if block.Ws !== nothing
        identity = identity * block.Ws
    end
    relu(out .+ identity)
end
