function sigmoid(x)
    x_arr = Float64.(x)
    1.0 ./ (1.0 .+ exp.(-x_arr))
end
