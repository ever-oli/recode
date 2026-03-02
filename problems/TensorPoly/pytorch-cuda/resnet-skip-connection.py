import torch


def compute_gradient_with_skip(gradients_F: list, x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    grad = torch.tensor(x, device=device)

    for F_grad in reversed(gradients_F):
        F_mat = torch.tensor(F_grad, device=device)
        dim = F_mat.shape[-1]
        grad = grad @ (torch.eye(dim, device=device) + F_mat)

    return grad


def compute_gradient_without_skip(gradients_F: list, x: torch.Tensor, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    grad = torch.tensor(x, device=device)

    for F_grad in reversed(gradients_F):
        F_mat = torch.tensor(F_grad, device=device)
        grad = grad @ F_mat

    return grad
