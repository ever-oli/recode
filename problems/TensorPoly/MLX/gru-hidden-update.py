import mlx.core as mx


def hidden_update(h_prev: mx.array, h_tilde: mx.array, z_t: mx.array) -> mx.array:
    keep_old = z_t * h_prev
    use_new = (1 - z_t) * h_tilde
    return keep_old + use_new
