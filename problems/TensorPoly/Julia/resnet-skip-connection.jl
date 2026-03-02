function compute_gradient_with_skip(gradients_F, x)
    grad = copy(x)
    for F_grad in reverse(gradients_F)
        F_mat = F_grad
        dim = size(F_mat, 2)
        grad = grad * (I + F_mat)
    end
    grad
end

function compute_gradient_without_skip(gradients_F, x)
    grad = copy(x)
    for F_grad in reverse(gradients_F)
        F_mat = F_grad
        grad = grad * F_mat
    end
    grad
end
