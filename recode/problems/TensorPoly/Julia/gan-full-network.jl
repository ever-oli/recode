sigmoid(x) = 1 ./ (1 .+ exp.(-clamp.(x, -500, 500)))

mutable struct GAN
    data_dim::Int
    noise_dim::Int
    G_W1
    G_b1
    G_W2
    G_b2
    D_W1
    D_b1
    D_W2
    D_b2
    D_W3
    D_b3
    d_lr::Float64
    g_lr::Float64
end

function GAN(data_dim::Int, noise_dim::Int)
    G_W1 = randn(noise_dim, 128) .* 0.02
    G_b1 = zeros(128)
    G_W2 = randn(128, data_dim) .* 0.02
    G_b2 = zeros(data_dim)

    D_W1 = randn(data_dim, 256) .* 0.02
    D_b1 = zeros(256)
    D_W2 = randn(256, 128) .* 0.02
    D_b2 = zeros(128)
    D_W3 = randn(128, 1) .* 0.02
    D_b3 = zeros(1)

    GAN(data_dim, noise_dim, G_W1, G_b1, G_W2, G_b2, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3, 0.001, 0.001)
end

function _generator_forward(model::GAN, z)
    h = max.(0, z * model.G_W1 .+ model.G_b1)
    tanh.(h * model.G_W2 .+ model.G_b2)
end

function _discriminator_forward(model::GAN, x)
    h1 = x * model.D_W1 .+ model.D_b1
    h1 = max.(0.2 .* h1, h1)
    h2 = h1 * model.D_W2 .+ model.D_b2
    h2 = max.(0.2 .* h2, h2)
    logits = h2 * model.D_W3 .+ model.D_b3
    vec(sigmoid(logits))
end

function generate(model::GAN, n::Int)
    z = randn(n, model.noise_dim)
    _generator_forward(model, z)
end

function discriminate(model::GAN, x)
    _discriminator_forward(model, x)
end

function train_step(model::GAN, real_data)
    batch_size = size(real_data, 1)
    eps = 1e-8
    fake_data = generate(model, batch_size)
    real_probs = discriminate(model, real_data)
    fake_probs = discriminate(model, fake_data)
    d_loss = -mean(log.(real_probs .+ eps) .+ log.(1 .- fake_probs .+ eps))
    g_loss = -mean(log.(fake_probs .+ eps))
    return (d_loss = d_loss, g_loss = g_loss)
end
