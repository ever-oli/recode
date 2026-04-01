import numpy as np


def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    batch_size = features.shape[0]
    x = features.reshape(batch_size, -1)

    def dense_relu(input_data, out_dim):
        in_dim = input_data.shape[1]
        limit = np.sqrt(2 / in_dim)
        w = np.random.randn(in_dim, out_dim) * limit
        b = np.zeros(out_dim)
        return np.maximum(0, input_data @ w + b)

    x = dense_relu(x, 4096)
    x = dense_relu(x, 4096)

    in_dim_final = x.shape[1]
    w_final = np.random.randn(in_dim_final, num_classes) * np.sqrt(2 / in_dim_final)
    b_final = np.zeros(num_classes)
    logits = x @ w_final + b_final
    return logits
