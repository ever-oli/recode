sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

function discriminator(x)
    input_dim = size(x, 2)
    W1 = randn(input_dim, 256) .* 0.02
    b1 = zeros(256)
    W2 = randn(256, 128) .* 0.02
    b2 = zeros(128)
    W3 = randn(128, 1) .* 0.02
    b3 = zeros(1)

    h1 = x * W1 .+ b1
    h1 = max.(0.2 .* h1, h1)
    h2 = h1 * W2 .+ b2
    h2 = max.(0.2 .* h2, h2)
    logits = h2 * W3 .+ b3
    sigmoid(logits)
end
