import mlx.core as mx


def bptt_single_step(dh_next: mx.array, h_t: mx.array, h_prev: mx.array, x_t: mx.array, W_hh: mx.array) -> tuple:
    dtanh = (1 - mx.square(h_t)) * dh_next
    dW_hh = mx.matmul(mx.transpose(dtanh), h_prev)
    dh_prev = mx.matmul(dtanh, W_hh)
    return dh_prev, dW_hh
