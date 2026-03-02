using Random

function create_nsp_examples(documents, num_examples::Int; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    examples = []
    while length(examples) < num_examples
        doc_idx = rand(1:length(documents))
        document = documents[doc_idx]

        if length(document) < 2
            continue
        end

        sent_idx = rand(1:(length(document) - 1))

        if rand() < 0.5
            push!(examples, (document[sent_idx], document[sent_idx + 1], 1))
        else
            if length(documents) > 1
                random_doc_idx = doc_idx
                while random_doc_idx == doc_idx
                    random_doc_idx = rand(1:length(documents))
                end
                random_document = documents[random_doc_idx]
            else
                random_document = document
            end
            random_sent_idx = rand(1:length(random_document))
            push!(examples, (document[sent_idx], random_document[random_sent_idx], 0))
        end
    end

    examples[1:num_examples]
end

mutable struct NSPHead
    W
    b
end

function NSPHead(hidden_size::Int)
    W = randn(hidden_size, 2) .* 0.02
    b = zeros(2)
    NSPHead(W, b)
end

function forward(head::NSPHead, cls_hidden)
    cls_hidden * head.W .+ head.b
end

softmax(x) = exp.(x .- maximum(x, dims=2)) ./ sum(exp.(x .- maximum(x, dims=2)), dims=2)
