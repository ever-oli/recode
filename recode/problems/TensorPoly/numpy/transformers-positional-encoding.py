import numpy as np


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    position = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(0, d_model, 2)
    div_term = np.exp(i * (-np.log(10000.0) / d_model))

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
