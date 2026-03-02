mutable struct WordPieceTokenizer
    vocab::Dict{String, Int}
    unk_token::String
    max_word_len::Int
end

function WordPieceTokenizer(vocab::Dict{String, Int}; unk_token::String="[UNK]", max_word_len::Int=100)
    WordPieceTokenizer(vocab, unk_token, max_word_len)
end

function tokenize(tokenizer::WordPieceTokenizer, text::String)
    tokens = String[]
    for word in split(lowercase(text))
        append!(tokens, tokenize_word(tokenizer, word))
    end
    tokens
end

function tokenize_word(tokenizer::WordPieceTokenizer, word::String)
    if length(word) > tokenizer.max_word_len
        return [tokenizer.unk_token]
    end

    output_tokens = String[]
    start = 1
    is_bad = false

    while start <= lastindex(word)
        end_idx = lastindex(word)
        cur_substr = nothing

        while start <= end_idx
            substr = word[start:end_idx]
            if start > 1
                substr = "##" * substr
            end
            if haskey(tokenizer.vocab, substr)
                cur_substr = substr
                break
            end
            end_idx -= 1
        end

        if cur_substr === nothing
            is_bad = true
            break
        end

        push!(output_tokens, cur_substr)
        start = end_idx + 1
    end

    if is_bad
        return [tokenizer.unk_token]
    end

    output_tokens
end
