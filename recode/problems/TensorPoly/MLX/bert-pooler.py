import mlx.core as mx


def tanh(x):
    return mx.tanh(x)


class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = mx.random.normal(shape=(hidden_size, hidden_size)) * 0.02
        self.b = mx.zeros((hidden_size,))

    def forward(self, hidden_states: mx.array) -> mx.array:
        cls_token_tensor = hidden_states[:, 0]
        pooled_output = mx.matmul(cls_token_tensor, self.W) + self.b
        return tanh(pooled_output)


class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = mx.random.normal(shape=(hidden_size, num_classes)) * 0.02
        self.bias = mx.zeros((num_classes,))

    def forward(self, hidden_states: mx.array, training: bool = True) -> mx.array:
        pooled_output = self.pooler.forward(hidden_states)
        if training:
            mask = (mx.random.uniform(shape=pooled_output.shape) > self.dropout_prob)
            pooled_output = (pooled_output * mask) / (1.0 - self.dropout_prob)
        return mx.matmul(pooled_output, self.classifier) + self.bias
