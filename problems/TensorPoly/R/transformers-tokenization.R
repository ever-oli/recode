SimpleTokenizer <- setRefClass(
  "SimpleTokenizer",
  fields = list(
    word_to_id = "list",
    id_to_word = "list",
    vocab_size = "numeric",
    pad_token = "character",
    unk_token = "character",
    bos_token = "character",
    eos_token = "character"
  ),
  methods = list(
    initialize = function() {
      word_to_id <<- list()
      id_to_word <<- list()
      vocab_size <<- 0
      pad_token <<- "<PAD>"
      unk_token <<- "<UNK>"
      bos_token <<- "<BOS>"
      eos_token <<- "<EOS>"
    },
    build_vocab = function(texts) {
      special_tokens <- c(pad_token, unk_token, bos_token, eos_token)
      for (idx in seq_along(special_tokens)) {
        token <- special_tokens[idx]
        word_to_id[[token]] <<- idx - 1
        id_to_word[[as.character(idx - 1)]] <<- token
      }

      unique_words <- unique(unlist(strsplit(texts, " ")))
      current_id <- length(special_tokens)
      for (word in sort(unique_words)) {
        if (is.null(word_to_id[[word]])) {
          word_to_id[[word]] <<- current_id
          id_to_word[[as.character(current_id)]] <<- word
          current_id <- current_id + 1
        }
      }
      vocab_size <<- length(word_to_id)
    },
    encode = function(text) {
      words <- unlist(strsplit(text, " "))
      sapply(words, function(word) {
        if (!is.null(word_to_id[[word]])) {
          word_to_id[[word]]
        } else {
          word_to_id[[unk_token]]
        }
      })
    },
    decode = function(ids) {
      words <- sapply(ids, function(token_id) {
        word <- id_to_word[[as.character(token_id)]]
        if (is.null(word)) unk_token else word
      })
      paste(words, collapse = " ")
    }
  )
)
