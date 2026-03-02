import torch


def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.as_tensor(param, device=device)
    grad = torch.as_tensor(grad, device=device)
    m = torch.as_tensor(m, device=device)
    v = torch.as_tensor(v, device=device)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    param_new = param - lr * m_hat / (torch.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
