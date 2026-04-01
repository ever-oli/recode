function vae_loss(x, x_recon, mu, log_var)
    recon_loss_per_sample = sum((x .- x_recon) .^ 2, dims=2)
    recon_loss = mean(recon_loss_per_sample)

    var = exp.(log_var)
    kl_per_sample = -0.5 .* sum(1 .+ log_var .- mu .^ 2 .- var, dims=2)
    kl_loss = mean(kl_per_sample)

    total_loss = recon_loss + kl_loss
    return (total = Float64(total_loss), recon = Float64(recon_loss), kl = Float64(kl_loss))
end
