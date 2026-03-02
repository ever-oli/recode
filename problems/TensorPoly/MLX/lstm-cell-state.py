import mlx.core as mx


def update_cell_state(C_prev: mx.array, f_t: mx.array, i_t: mx.array, c_tilde: mx.array) -> mx.array:
    return f_t * C_prev + i_t * c_tilde
