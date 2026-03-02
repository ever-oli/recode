function compute_gradient_norm_decay(T::Int, W_hh)
    spectral_norm = opnorm(W_hh, 2)
    norms = Float64[]
    current_norm = 1.0
    push!(norms, current_norm)

    for _ in 2:T
        current_norm *= spectral_norm
        push!(norms, current_norm)
    end

    norms
end
