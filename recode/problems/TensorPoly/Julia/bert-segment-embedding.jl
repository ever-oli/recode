mutable struct BertEmbeddings
    hidden_size::Int
    token_embeddings
    position_embeddings
    segment_embeddings
end

function BertEmbeddings(vocab_size::Int, max_position::Int, hidden_size::Int)
    token_embeddings = randn(vocab_size, hidden_size) .* 0.02
    position_embeddings = randn(max_position, hidden_size) .* 0.02
    segment_embeddings = randn(2, hidden_size) .* 0.02
    BertEmbeddings(hidden_size, token_embeddings, position_embeddings, segment_embeddings)
end

function forward(emb::BertEmbeddings, token_ids, segment_ids)
    tok_emb = emb.token_embeddings[token_ids .+ 1, :]
    seq_len = size(token_ids, 2)
    positions = 1:seq_len
    pos_emb = emb.position_embeddings[positions, :]
    seg_emb = emb.segment_embeddings[segment_ids .+ 1, :]
    tok_emb .+ pos_emb .+ seg_emb
end
