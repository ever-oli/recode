function vae_encoder(x, latent_dim::Int)
    input_dim = size(x, 2)
    hidden_dim = 256

    w_h = randn(input_dim, hidden_dim) .* 0.01
    b_h = zeros(hidden_dim)
    h = max.(0, x * w_h .+ b_h)

    w_mu = randn(hidden_dim, latent_dim) .* 0.01
    b_mu = zeros(latent_dim)
    mu = h * w_mu .+ b_mu

    w_log_var = randn(hidden_dim, latent_dim) .* 0.01
    b_log_var = zeros(latent_dim)
    log_var = h * w_log_var .+ b_log_var

    return (mu = mu, log_var = log_var)
end
