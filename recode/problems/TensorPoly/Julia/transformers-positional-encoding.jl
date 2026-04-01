function positional_encoding(seq_length::Int, d_model::Int)
    position = reshape(0:(seq_length - 1), :, 1)
    i = 0:2:(d_model - 1)
    div_term = exp.(i .* (-log(10000.0) / d_model))

    pe = zeros(seq_length, d_model)
    pe[:, 1:2:end] .= sin.(position * div_term')
    if d_model > 1
        pe[:, 2:2:end] .= cos.(position * div_term[1:length(2:2:end)]')
    end
    pe
end
