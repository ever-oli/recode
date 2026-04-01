function random_crop(image, crop_size::Int=224)
    h = size(image, 1)
    w = size(image, 2)
    top = rand(1:(h - crop_size + 1))
    left = rand(1:(w - crop_size + 1))
    return image[top:(top + crop_size - 1), left:(left + crop_size - 1), :]
end


function random_horizontal_flip(image, p::Float64=0.5)
    if rand() < p
        return image[:, end:-1:1, :]
    end
    return image
end
