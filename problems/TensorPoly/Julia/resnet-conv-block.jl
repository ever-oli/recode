relu(x) = max.(0, x)

mutable struct ConvBlock
    in_channels::Int
    out_channels::Int
    W1
    W2
    Ws
end

function ConvBlock(in_channels::Int, out_channels::Int)
    W1 = randn(in_channels, out_channels) .* 0.01
    W2 = randn(out_channels, out_channels) .* 0.01
    Ws = randn(in_channels, out_channels) .* 0.01
    ConvBlock(in_channels, out_channels, W1, W2, Ws)
end

function forward(block::ConvBlock, x)
    main = relu(x * block.W1)
    main = main * block.W2
    shortcut = x * block.Ws
    relu(main .+ shortcut)
end
