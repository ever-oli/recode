import numpy as np


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    Q_trans = Q_heads.transpose(0, 2, 1, 3)
    K_trans = K_heads.transpose(0, 2, 1, 3)
    V_trans = V_heads.transpose(0, 2, 1, 3)

    scores = np.matmul(Q_trans, K_trans.transpose(0, 1, 3, 2))
    scaled_scores = scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis=-1)
    head_outputs = np.matmul(attention_weights, V_trans)

    head_outputs_trans = head_outputs.transpose(0, 2, 1, 3)
    concatenated = head_outputs_trans.reshape(batch_size, seq_len, d_model)
    output = np.matmul(concatenated, W_o)
    return output
