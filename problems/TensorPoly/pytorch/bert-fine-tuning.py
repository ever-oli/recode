import torch
from typing import List


class MockBertEncoder:
    """Simulated BERT encoder with 12 layers."""

    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = [torch.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers

    def freeze_layers(self, layer_indices: List[int]):
        for idx in layer_indices:
            if 0 <= idx < self.num_layers:
                self.layer_frozen[idx] = True

    def unfreeze_all(self):
        self.layer_frozen = [False] * self.num_layers

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings
        for layer in self.layers:
            x = torch.matmul(x, layer) + x
        return x


class BertForSequenceClassification:
    """BERT with sequence-level classification head (e.g. Sentiment)."""

    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = torch.randn(hidden_size, num_labels) * 0.02
        self.bias = torch.zeros(num_labels)
        self.freeze_bert = freeze_bert

        if freeze_bert:
            self.encoder.freeze_layers(list(range(12)))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder.forward(embeddings)
        cls_representation = hidden_states[:, 0, :]
        logits = torch.matmul(cls_representation, self.classifier) + self.bias
        return logits


class BertForTokenClassification:
    """BERT with token-level classification (e.g. NER, POS tagging)."""

    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder = MockBertEncoder(hidden_size)
        self.classifier = torch.randn(hidden_size, num_labels) * 0.02
        self.bias = torch.zeros(num_labels)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder.forward(embeddings)
        return torch.matmul(hidden_states, self.classifier) + self.bias
