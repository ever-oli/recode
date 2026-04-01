import mlx.core as mx


def vgg_classifier(features: mx.array, num_classes: int = 1000) -> mx.array:
    batch_size = features.shape[0]
    x = mx.reshape(features, (batch_size, -1))

    def dense_relu(input_data: mx.array, out_dim: int) -> mx.array:
        in_dim = input_data.shape[1]
        limit = mx.sqrt(mx.array(2.0 / in_dim))
        w = mx.random.normal(shape=(in_dim, out_dim)) * limit
        b = mx.zeros((out_dim,))
        return mx.maximum(0, mx.matmul(input_data, w) + b)

    x = dense_relu(x, 4096)
    x = dense_relu(x, 4096)

    in_dim_final = x.shape[1]
    w_final = mx.random.normal(shape=(in_dim_final, num_classes)) * mx.sqrt(mx.array(2.0 / in_dim_final))
    b_final = mx.zeros((num_classes,))
    return mx.matmul(x, w_final) + b_final
