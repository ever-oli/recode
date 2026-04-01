tanh_act <- function(x) {
  tanh(x)
}

BertPooler <- setRefClass(
  "BertPooler",
  fields = list(
    hidden_size = "numeric",
    W = "matrix",
    b = "numeric"
  ),
  methods = list(
    initialize = function(hidden_size) {
      hidden_size <<- hidden_size
      W <<- matrix(rnorm(hidden_size * hidden_size, sd = 0.02), nrow = hidden_size, ncol = hidden_size)
      b <<- numeric(hidden_size)
    },
    forward = function(hidden_states) {
      cls_token_tensor <- hidden_states[, 1, ]
      pooled_output <- cls_token_tensor %*% W + b
      tanh_act(pooled_output)
    }
  )
)

SequenceClassifier <- setRefClass(
  "SequenceClassifier",
  fields = list(
    pooler = "BertPooler",
    dropout_prob = "numeric",
    classifier = "matrix",
    bias = "numeric"
  ),
  methods = list(
    initialize = function(hidden_size, num_classes, dropout_prob = 0.1) {
      pooler <<- BertPooler$new(hidden_size)
      dropout_prob <<- dropout_prob
      classifier <<- matrix(rnorm(hidden_size * num_classes, sd = 0.02), nrow = hidden_size, ncol = num_classes)
      bias <<- numeric(num_classes)
    },
    forward = function(hidden_states, training = TRUE) {
      pooled_output <- pooler$forward(hidden_states)
      if (training) {
        mask <- matrix(runif(length(pooled_output)) > dropout_prob, nrow = nrow(pooled_output))
        pooled_output <- (pooled_output * mask) / (1.0 - dropout_prob)
      }
      pooled_output %*% classifier + bias
    }
  )
)
