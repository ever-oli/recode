mutable struct SimpleTokenizer
    word_to_id::Dict{String, Int}
    id_to_word::Dict{Int, String}
    vocab_size::Int
    pad_token::String
    unk_token::String
    bos_token::String
    eos_token::String
end

function SimpleTokenizer()
    SimpleTokenizer(Dict{String, Int}(), Dict{Int, String}(), 0, "<PAD>", "<UNK>", "<BOS>", "<EOS>")
end

function build_vocab!(tokenizer::SimpleTokenizer, texts::Vector{String})
    special_tokens = [tokenizer.pad_token, tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token]
    for (idx, token) in enumerate(special_tokens)
        tokenizer.word_to_id[token] = idx - 1
        tokenizer.id_to_word[idx - 1] = token
    end

    unique_words = Set{String}()
    for text in texts
        for word in split(text)
            push!(unique_words, word)
        end
    end

    current_id = length(special_tokens)
    for word in sort(collect(unique_words))
        if !haskey(tokenizer.word_to_id, word)
            tokenizer.word_to_id[word] = current_id
            tokenizer.id_to_word[current_id] = word
            current_id += 1
        end
    end

    tokenizer.vocab_size = length(tokenizer.word_to_id)
end

function encode(tokenizer::SimpleTokenizer, text::String)
    words = split(text)
    [get(tokenizer.word_to_id, word, tokenizer.word_to_id[tokenizer.unk_token]) for word in words]
end

function decode(tokenizer::SimpleTokenizer, ids::Vector{Int})
    words = [get(tokenizer.id_to_word, token_id, tokenizer.unk_token) for token_id in ids]
    join(words, " ")
end
