import mlx.core as mx


def compute_gradient_with_skip(gradients_F: list, x: mx.array) -> mx.array:
    grad = mx.array(x)

    for F_grad in reversed(gradients_F):
        F_mat = mx.array(F_grad)
        dim = F_mat.shape[-1]
        grad = mx.matmul(grad, mx.eye(dim) + F_mat)

    return grad


def compute_gradient_without_skip(gradients_F: list, x: mx.array) -> mx.array:
    grad = mx.array(x)

    for F_grad in reversed(gradients_F):
        F_mat = mx.array(F_grad)
        grad = mx.matmul(grad, F_mat)

    return grad
