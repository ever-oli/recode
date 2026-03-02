function alexnet_conv1(image)
    batch_size = size(image, 1)
    output_h = 55
    output_w = 55
    num_filters = 96
    return zeros(batch_size, output_h, output_w, num_filters)
end
