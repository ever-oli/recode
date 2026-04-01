import mlx.core as mx


def vgg16(x: mx.array, num_classes: int = 1000) -> mx.array:
    vgg16_config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]

    features = vgg_features(x, vgg16_config)
    return vgg_classifier(features, num_classes)
