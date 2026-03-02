relu(x) = max.(0, x)

mutable struct BasicBlock
    in_ch::Int
    out_ch::Int
    downsample::Bool
    W1
    W2
    W_proj
end

function BasicBlock(in_ch::Int, out_ch::Int; downsample::Bool=false)
    W1 = randn(in_ch, out_ch) .* 0.01
    W2 = randn(out_ch, out_ch) .* 0.01
    W_proj = (in_ch != out_ch || downsample) ? randn(in_ch, out_ch) .* 0.01 : nothing
    BasicBlock(in_ch, out_ch, downsample, W1, W2, W_proj)
end

function forward(block::BasicBlock, x)
    identity = x
    out = relu(x * block.W1)
    out = out * block.W2
    if block.W_proj !== nothing
        identity = identity * block.W_proj
    end
    relu(out .+ identity)
end

mutable struct ResNet18
    conv1
    layer1
    layer2
    layer3
    layer4
    fc
end

function ResNet18(num_classes::Int=10)
    conv1 = randn(3, 64) .* 0.01
    layer1 = [BasicBlock(64, 64, downsample=false), BasicBlock(64, 64, downsample=false)]
    layer2 = [BasicBlock(64, 128, downsample=true), BasicBlock(128, 128, downsample=false)]
    layer3 = [BasicBlock(128, 256, downsample=true), BasicBlock(256, 256, downsample=false)]
    layer4 = [BasicBlock(256, 512, downsample=true), BasicBlock(512, 512, downsample=false)]
    fc = randn(512, num_classes) .* 0.01
    ResNet18(conv1, layer1, layer2, layer3, layer4, fc)
end

function forward(model::ResNet18, x)
    out = relu(x * model.conv1)
    for block in model.layer1
        out = forward(block, out)
    end
    for block in model.layer2
        out = forward(block, out)
    end
    for block in model.layer3
        out = forward(block, out)
    end
    for block in model.layer4
        out = forward(block, out)
    end
    out * model.fc
end
