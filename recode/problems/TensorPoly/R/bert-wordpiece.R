WordPieceTokenizer <- setRefClass(
  "WordPieceTokenizer",
  fields = list(
    vocab = "list",
    unk_token = "character",
    max_word_len = "numeric"
  ),
  methods = list(
    initialize = function(vocab, unk_token = "[UNK]", max_word_len = 100) {
      vocab <<- vocab
      unk_token <<- unk_token
      max_word_len <<- max_word_len
    },
    tokenize = function(text) {
      tokens <- character(0)
      words <- unlist(strsplit(tolower(text), " "))
      for (word in words) {
        word_tokens <- .self$.tokenize_word(word)
        tokens <- c(tokens, word_tokens)
      }
      tokens
    },
    .tokenize_word = function(word) {
      if (nchar(word) > max_word_len) {
        return(c(unk_token))
      }

      output_tokens <- character(0)
      start <- 1
      is_bad <- FALSE

      while (start <= nchar(word)) {
        end <- nchar(word)
        cur_substr <- NULL

        while (start <= end) {
          substr <- substr(word, start, end)
          if (start > 1) {
            substr <- paste0("##", substr)
          }

          if (!is.null(vocab[[substr]])) {
            cur_substr <- substr
            break
          }
          end <- end - 1
        }

        if (is.null(cur_substr)) {
          is_bad <- TRUE
          break
        }

        output_tokens <- c(output_tokens, cur_substr)
        start <- end + 1
      }

      if (is_bad) {
        return(c(unk_token))
      }

      output_tokens
    }
  )
)
