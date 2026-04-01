import torch


def compute_gradient_with_skip(gradients_F: list, x: torch.Tensor) -> torch.Tensor:
    grad = torch.tensor(x, copy=True)

    for F_grad in reversed(gradients_F):
        F_mat = torch.tensor(F_grad)
        dim = F_mat.shape[-1]
        grad = grad @ (torch.eye(dim) + F_mat)

    return grad


def compute_gradient_without_skip(gradients_F: list, x: torch.Tensor) -> torch.Tensor:
    grad = torch.tensor(x, copy=True)

    for F_grad in reversed(gradients_F):
        F_mat = torch.tensor(F_grad)
        grad = grad @ F_mat

    return grad
