import numpy as np


def vgg16(x: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    vgg16_config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]

    features = vgg_features(x, vgg16_config)
    return vgg_classifier(features, num_classes)


def conv_relu(x, out_channels):
    _, _, _, C = x.shape
    W_weights = np.random.randn(C, out_channels) * 0.1
    x = x @ W_weights
    return np.maximum(0, x)


def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).max(axis=(2, 4))


def vgg_features(x: np.ndarray, config: list) -> np.ndarray:
    out = x

    for layer in config:
        if isinstance(layer, int):
            out = conv_relu(out, layer)
        elif layer == "M":
            out = maxpool_2x2(out)

    return out


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
