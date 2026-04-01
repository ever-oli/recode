import mlx.core as mx


def unet_output(features: mx.array, num_classes: int) -> mx.array:
    batch, H, W, _ = features.shape
    return mx.zeros((batch, H, W, num_classes))
