MockBertEncoder <- setRefClass(
  "MockBertEncoder",
  fields = list(
    hidden_size = "numeric",
    num_layers = "numeric",
    layers = "list",
    layer_frozen = "logical"
  ),
  methods = list(
    initialize = function(hidden_size = 768, num_layers = 12) {
      hidden_size <<- hidden_size
      num_layers <<- num_layers
      layers <<- lapply(seq_len(num_layers), function(i) matrix(rnorm(hidden_size * hidden_size, sd = 0.01), nrow = hidden_size, ncol = hidden_size))
      layer_frozen <<- rep(FALSE, num_layers)
    },
    freeze_layers = function(layer_indices) {
      for (idx in layer_indices) {
        if (idx >= 1 && idx <= num_layers) {
          layer_frozen[idx] <<- TRUE
        }
      }
    },
    unfreeze_all = function() {
      layer_frozen <<- rep(FALSE, num_layers)
    },
    forward = function(embeddings) {
      x <- embeddings
      for (layer in layers) {
        x <- x %*% layer + x
      }
      x
    }
  )
)

BertForSequenceClassification <- setRefClass(
  "BertForSequenceClassification",
  fields = list(
    encoder = "MockBertEncoder",
    classifier = "matrix",
    bias = "numeric",
    freeze_bert = "logical"
  ),
  methods = list(
    initialize = function(hidden_size, num_labels, freeze_bert = FALSE) {
      encoder <<- MockBertEncoder$new(hidden_size)
      classifier <<- matrix(rnorm(hidden_size * num_labels, sd = 0.02), nrow = hidden_size, ncol = num_labels)
      bias <<- numeric(num_labels)
      freeze_bert <<- freeze_bert
      if (freeze_bert) {
        encoder$freeze_layers(1:12)
      }
    },
    forward = function(embeddings) {
      hidden_states <- encoder$forward(embeddings)
      cls_representation <- hidden_states[, 1, ]
      cls_representation %*% classifier + bias
    }
  )
)

BertForTokenClassification <- setRefClass(
  "BertForTokenClassification",
  fields = list(
    encoder = "MockBertEncoder",
    classifier = "matrix",
    bias = "numeric"
  ),
  methods = list(
    initialize = function(hidden_size, num_labels) {
      encoder <<- MockBertEncoder$new(hidden_size)
      classifier <<- matrix(rnorm(hidden_size * num_labels, sd = 0.02), nrow = hidden_size, ncol = num_labels)
      bias <<- numeric(num_labels)
    },
    forward = function(embeddings) {
      hidden_states <- encoder$forward(embeddings)
      hidden_states %*% classifier + bias
    }
  )
)
