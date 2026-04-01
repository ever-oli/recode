create_nsp_examples <- function(documents, num_examples, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  examples <- list()
  while (length(examples) < num_examples) {
    doc_idx <- sample(seq_along(documents), 1)
    document <- documents[[doc_idx]]

    if (length(document) < 2) {
      next
    }

    sent_idx <- sample(1:(length(document) - 1), 1)
    if (runif(1) < 0.5) {
      examples[[length(examples) + 1]] <- list(document[[sent_idx]], document[[sent_idx + 1]], 1)
    } else {
      if (length(documents) > 1) {
        random_doc_idx <- doc_idx
        while (random_doc_idx == doc_idx) {
          random_doc_idx <- sample(seq_along(documents), 1)
        }
        random_document <- documents[[random_doc_idx]]
      } else {
        random_document <- document
      }
      random_sent_idx <- sample(seq_along(random_document), 1)
      examples[[length(examples) + 1]] <- list(document[[sent_idx]], random_document[[random_sent_idx]], 0)
    }
  }

  examples[1:num_examples]
}

NSPHead <- setRefClass(
  "NSPHead",
  fields = list(
    W = "matrix",
    b = "numeric"
  ),
  methods = list(
    initialize = function(hidden_size) {
      W <<- matrix(rnorm(hidden_size * 2, sd = 0.02), nrow = hidden_size, ncol = 2)
      b <<- numeric(2)
    },
    forward = function(cls_hidden) {
      cls_hidden %*% W + b
    }
  )
)

softmax <- function(x) {
  exp_x <- exp(x - apply(x, 2, max))
  exp_x / rowSums(exp_x)
}
