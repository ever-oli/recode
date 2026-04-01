relu <- function(x) {
  pmax(0, x)
}

BasicBlock <- setRefClass(
  "BasicBlock",
  fields = list(
    in_ch = "numeric",
    out_ch = "numeric",
    downsample = "logical",
    W1 = "matrix",
    W2 = "matrix",
    W_proj = "matrix"
  ),
  methods = list(
    initialize = function(in_ch, out_ch, downsample = FALSE) {
      in_ch <<- in_ch
      out_ch <<- out_ch
      downsample <<- downsample
      W1 <<- matrix(rnorm(in_ch * out_ch, sd = 0.01), nrow = in_ch, ncol = out_ch)
      W2 <<- matrix(rnorm(out_ch * out_ch, sd = 0.01), nrow = out_ch, ncol = out_ch)
      W_proj <<- if (in_ch != out_ch || downsample) matrix(rnorm(in_ch * out_ch, sd = 0.01), nrow = in_ch, ncol = out_ch) else NULL
    },
    forward = function(x) {
      identity <- x
      out <- relu(x %*% W1)
      out <- out %*% W2
      if (!is.null(W_proj)) {
        identity <- identity %*% W_proj
      }
      relu(out + identity)
    }
  )
)

ResNet18 <- setRefClass(
  "ResNet18",
  fields = list(
    conv1 = "matrix",
    layer1 = "list",
    layer2 = "list",
    layer3 = "list",
    layer4 = "list",
    fc = "matrix"
  ),
  methods = list(
    initialize = function(num_classes = 10) {
      conv1 <<- matrix(rnorm(3 * 64, sd = 0.01), nrow = 3, ncol = 64)
      layer1 <<- list(BasicBlock$new(64, 64, FALSE), BasicBlock$new(64, 64, FALSE))
      layer2 <<- list(BasicBlock$new(64, 128, TRUE), BasicBlock$new(128, 128, FALSE))
      layer3 <<- list(BasicBlock$new(128, 256, TRUE), BasicBlock$new(256, 256, FALSE))
      layer4 <<- list(BasicBlock$new(256, 512, TRUE), BasicBlock$new(512, 512, FALSE))
      fc <<- matrix(rnorm(512 * num_classes, sd = 0.01), nrow = 512, ncol = num_classes)
    },
    forward = function(x) {
      out <- relu(x %*% conv1)
      for (block in layer1) out <- block$forward(out)
      for (block in layer2) out <- block$forward(out)
      for (block in layer3) out <- block$forward(out)
      for (block in layer4) out <- block$forward(out)
      out %*% fc
    }
  )
)
