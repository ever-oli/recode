mutable struct MockBertEncoder
    hidden_size::Int
    num_layers::Int
    layers::Vector
    layer_frozen::Vector{Bool}
end

function MockBertEncoder(hidden_size::Int=768, num_layers::Int=12)
    layers = [randn(hidden_size, hidden_size) .* 0.01 for _ in 1:num_layers]
    layer_frozen = fill(false, num_layers)
    MockBertEncoder(hidden_size, num_layers, layers, layer_frozen)
end

function freeze_layers!(encoder::MockBertEncoder, layer_indices)
    for idx in layer_indices
        if 1 <= idx <= encoder.num_layers
            encoder.layer_frozen[idx] = true
        end
    end
end

function unfreeze_all!(encoder::MockBertEncoder)
    encoder.layer_frozen .= false
end

function forward(encoder::MockBertEncoder, embeddings)
    x = embeddings
    for layer in encoder.layers
        x = x * layer .+ x
    end
    x
end

mutable struct BertForSequenceClassification
    encoder::MockBertEncoder
    classifier
    bias
    freeze_bert::Bool
end

function BertForSequenceClassification(hidden_size::Int, num_labels::Int; freeze_bert::Bool=false)
    encoder = MockBertEncoder(hidden_size)
    classifier = randn(hidden_size, num_labels) .* 0.02
    bias = zeros(num_labels)
    model = BertForSequenceClassification(encoder, classifier, bias, freeze_bert)
    if freeze_bert
        freeze_layers!(model.encoder, 1:12)
    end
    model
end

function forward(model::BertForSequenceClassification, embeddings)
    hidden_states = forward(model.encoder, embeddings)
    cls_representation = hidden_states[:, 1, :]
    cls_representation * model.classifier .+ model.bias
end

mutable struct BertForTokenClassification
    encoder::MockBertEncoder
    classifier
    bias
end

function BertForTokenClassification(hidden_size::Int, num_labels::Int)
    encoder = MockBertEncoder(hidden_size)
    classifier = randn(hidden_size, num_labels) .* 0.02
    bias = zeros(num_labels)
    BertForTokenClassification(encoder, classifier, bias)
end

function forward(model::BertForTokenClassification, embeddings)
    hidden_states = forward(model.encoder, embeddings)
    hidden_states * model.classifier .+ model.bias
end
