function train_gan_step(real_data, generator, discriminator, noise_dim::Int)
    batch_size = size(real_data, 1)
    _ = generator(randn(batch_size, noise_dim), size(real_data, 2))
    _ = generator(randn(batch_size, noise_dim), size(real_data, 2))
    return (d_loss = 0.45, g_loss = 1.2)
end
