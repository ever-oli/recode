function vgg_maxpool(x)
    batch, h, w, c = size(x)
    reshaped = reshape(x, batch, div(h, 2), 2, div(w, 2), 2, c)
    maximum(reshaped, dims=(3, 5))
end
