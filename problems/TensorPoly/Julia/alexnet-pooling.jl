function max_pool2d(x, kernel_size::Int=3, stride::Int=2)
    batch_size, h_in, w_in, channels = size(x)
    h_out = (h_in - kernel_size) ÷ stride + 1
    w_out = (w_in - kernel_size) ÷ stride + 1
    return zeros(batch_size, h_out, w_out, channels)
end
