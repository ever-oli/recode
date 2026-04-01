using Random

function apply_mlm_mask(token_ids, vocab_size::Int; mask_token_id::Int=103, mask_prob::Float64=0.15, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    masked_ids = copy(token_ids)
    labels = fill(-100, size(token_ids))

    mask_eligible = .!(token_ids .== 101 .| token_ids .== 102 .| token_ids .== 0)
    probability_matrix = rand(size(token_ids))
    mask_indices = (probability_matrix .< mask_prob) .& mask_eligible

    labels[mask_indices] = token_ids[mask_indices]

    random_dispatch = rand(size(token_ids))
    indices_replaced = mask_indices .& (random_dispatch .< 0.8)
    masked_ids[indices_replaced] .= mask_token_id

    indices_random = mask_indices .& (random_dispatch .>= 0.8) .& (random_dispatch .< 0.9)
    masked_ids[indices_random] .= rand(0:(vocab_size - 1), sum(indices_random))

    return (masked_ids = masked_ids, labels = labels, mask_indices = mask_indices)
end

mutable struct MLMHead
    hidden_size::Int
    vocab_size::Int
    W
    b
end

function MLMHead(hidden_size::Int, vocab_size::Int)
    W = randn(hidden_size, vocab_size) .* 0.02
    b = zeros(vocab_size)
    MLMHead(hidden_size, vocab_size, W, b)
end

function forward(head::MLMHead, hidden_states)
    hidden_states * head.W .+ head.b
end
