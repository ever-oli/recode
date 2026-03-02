import mlx.core as mx


def positional_encoding(seq_length: int, d_model: int) -> mx.array:
    position = mx.arange(seq_length)[:, None]
    i = mx.arange(0, d_model, 2)
    div_term = mx.exp(i * (-mx.log(mx.array(10000.0)) / d_model))

    pe = mx.zeros((seq_length, d_model))
    pe_even = mx.sin(position * div_term)
    pe_odd = mx.cos(position * div_term)
    pe = mx.concatenate([pe_even, pe_odd], axis=1)
    return pe[:, :d_model]
