function dropout(x, p::Float64=0.5, training::Bool=true)
    if !training || p == 0
        return x
    end
    mask = rand(size(x)) .< (1 - p)
    return (x .* mask) ./ (1 - p)
end
