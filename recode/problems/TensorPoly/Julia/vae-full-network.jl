mutable struct VAE
    input_dim::Int
    latent_dim::Int
    hidden_dim::Int
    w_enc
    b_enc
    w_mu
    b_mu
    w_log_var
    b_log_var
    w_dec_h
    b_dec_h
    w_dec_out
    b_dec_out
end

function VAE(input_dim::Int, latent_dim::Int)
    hidden_dim = 256
    w_enc = randn(input_dim, hidden_dim) .* 0.01
    b_enc = zeros(hidden_dim)

    w_mu = randn(hidden_dim, latent_dim) .* 0.01
    b_mu = zeros(latent_dim)
    w_log_var = randn(hidden_dim, latent_dim) .* 0.01
    b_log_var = zeros(latent_dim)

    w_dec_h = randn(latent_dim, hidden_dim) .* 0.01
    b_dec_h = zeros(hidden_dim)
    w_dec_out = randn(hidden_dim, input_dim) .* 0.01
    b_dec_out = zeros(input_dim)

    VAE(input_dim, latent_dim, hidden_dim, w_enc, b_enc, w_mu, b_mu, w_log_var, b_log_var, w_dec_h, b_dec_h, w_dec_out, b_dec_out)
end

function forward(model::VAE, x)
    h_enc = max.(0, x * model.w_enc .+ model.b_enc)
    mu = h_enc * model.w_mu .+ model.b_mu
    log_var = h_enc * model.w_log_var .+ model.b_log_var

    std = exp.(0.5 .* log_var)
    eps = randn(size(mu))
    z = mu .+ std .* eps

    h_dec = max.(0, z * model.w_dec_h .+ model.b_dec_h)
    logits = h_dec * model.w_dec_out .+ model.b_dec_out
    x_recon = 1.0 ./ (1.0 .+ exp.(-logits))

    return (x_recon = x_recon, mu = mu, log_var = log_var)
end

function generate(model::VAE, n_samples::Int)
    z = randn(n_samples, model.latent_dim)
    h_dec = max.(0, z * model.w_dec_h .+ model.b_dec_h)
    logits = h_dec * model.w_dec_out .+ model.b_dec_out
    1.0 ./ (1.0 .+ exp.(-logits))
end
