function local_response_normalization(x, k::Float64=2, n::Int=5, alpha::Float64=1e-4, beta::Float64=0.75)
    batch_size, h, w, c = size(x)
    squared_x = x .^ 2
    pad = n ÷ 2
    padded_sq = zeros(batch_size, h, w, c + 2 * pad)
    padded_sq[:, :, :, (pad + 1):(pad + c)] .= squared_x

    sum_sq = zeros(batch_size, h, w, c)
    for i in 1:n
        sum_sq .+= padded_sq[:, :, :, i:(i + c - 1)]
    end

    scale = (k .+ alpha .* sum_sq) .^ beta
    return x ./ scale
end
