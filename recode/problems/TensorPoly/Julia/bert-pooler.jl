tanh_act(x) = tanh.(x)

mutable struct BertPooler
    hidden_size::Int
    W
    b
end

function BertPooler(hidden_size::Int)
    W = randn(hidden_size, hidden_size) .* 0.02
    b = zeros(hidden_size)
    BertPooler(hidden_size, W, b)
end

function forward(pooler::BertPooler, hidden_states)
    cls_token_tensor = hidden_states[:, 1, :]
    pooled_output = cls_token_tensor * pooler.W .+ pooler.b
    tanh_act(pooled_output)
end

mutable struct SequenceClassifier
    pooler::BertPooler
    dropout_prob::Float64
    classifier
    bias
end

function SequenceClassifier(hidden_size::Int, num_classes::Int; dropout_prob::Float64=0.1)
    pooler = BertPooler(hidden_size)
    classifier = randn(hidden_size, num_classes) .* 0.02
    bias = zeros(num_classes)
    SequenceClassifier(pooler, dropout_prob, classifier, bias)
end

function forward(model::SequenceClassifier, hidden_states; training::Bool=true)
    pooled_output = forward(model.pooler, hidden_states)
    if training
        mask = rand(size(pooled_output)) .> model.dropout_prob
        pooled_output = (pooled_output .* mask) ./ (1.0 - model.dropout_prob)
    end
    pooled_output * model.classifier .+ model.bias
end
