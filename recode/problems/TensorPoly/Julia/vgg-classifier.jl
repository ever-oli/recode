function vgg_classifier(features, num_classes::Int=1000)
    batch_size = size(features, 1)
    x = reshape(features, batch_size, :)

    function dense_relu(input_data, out_dim)
        in_dim = size(input_data, 2)
        limit = sqrt(2 / in_dim)
        w = randn(in_dim, out_dim) .* limit
        b = zeros(out_dim)
        max.(0, input_data * w .+ b)
    end

    x = dense_relu(x, 4096)
    x = dense_relu(x, 4096)

    in_dim_final = size(x, 2)
    w_final = randn(in_dim_final, num_classes) .* sqrt(2 / in_dim_final)
    b_final = zeros(num_classes)
    x * w_final .+ b_final
end
