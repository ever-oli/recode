import numpy as np


def tanh(x):
    return np.tanh(x)


class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        cls_token_tensor = hidden_states[:, 0]
        pooled_output = np.dot(cls_token_tensor, self.W) + self.b
        return tanh(pooled_output)


class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
        self.bias = np.zeros(num_classes)

    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        pooled_output = self.pooler.forward(hidden_states)
        if training:
            mask = (np.random.rand(*pooled_output.shape) > self.dropout_prob)
            pooled_output = (pooled_output * mask) / (1.0 - self.dropout_prob)
        return np.dot(pooled_output, self.classifier) + self.bias
