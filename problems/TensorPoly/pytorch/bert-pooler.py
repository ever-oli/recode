import torch


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = torch.randn(hidden_size, hidden_size) * 0.02
        self.b = torch.zeros(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token_tensor = hidden_states[:, 0]
        pooled_output = torch.matmul(cls_token_tensor, self.W) + self.b
        return tanh(pooled_output)


class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier = torch.randn(hidden_size, num_classes) * 0.02
        self.bias = torch.zeros(num_classes)

    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> torch.Tensor:
        pooled_output = self.pooler.forward(hidden_states)
        if training:
            mask = (torch.rand_like(pooled_output) > self.dropout_prob)
            pooled_output = (pooled_output * mask) / (1.0 - self.dropout_prob)
        return torch.matmul(pooled_output, self.classifier) + self.bias
