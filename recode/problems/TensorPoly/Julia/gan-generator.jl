function generator(z, output_dim::Int)
    noise_dim = size(z, 2)
    W1 = randn(noise_dim, 128) .* 0.02
    b1 = zeros(128)
    W2 = randn(128, output_dim) .* 0.02
    b2 = zeros(output_dim)

    h1 = max.(0, z * W1 .+ b1)
    tanh.(h1 * W2 .+ b2)
end
