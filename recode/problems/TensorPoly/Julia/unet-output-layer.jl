function unet_output(features, num_classes::Int)
    batch, H, W, _ = size(features)
    return zeros(batch, H, W, num_classes)
end
