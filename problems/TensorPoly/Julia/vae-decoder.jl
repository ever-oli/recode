function vae_decoder(z, output_dim::Int)
    latent_dim = size(z, 2)
    hidden_dim = 256

    w_h = randn(latent_dim, hidden_dim) .* 0.01
    b_h = zeros(hidden_dim)
    h = max.(0, z * w_h .+ b_h)

    w_out = randn(hidden_dim, output_dim) .* 0.01
    b_out = zeros(output_dim)
    logits = h * w_out .+ b_out

    return 1.0 ./ (1.0 .+ exp.(-logits))
end
