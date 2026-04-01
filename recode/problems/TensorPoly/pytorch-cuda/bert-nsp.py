import torch
from typing import List, Tuple
import random


def create_nsp_examples(documents: List[List[str]], num_examples: int, seed: int = None) -> List[Tuple[str, str, int]]:
    if seed is not None:
        random.seed(seed)

    examples = []
    while len(examples) < num_examples:
        doc_idx = random.randint(0, len(documents) - 1)
        document = documents[doc_idx]

        if len(document) < 2:
            continue

        sent_idx = random.randint(0, len(document) - 2)

        if random.random() < 0.5:
            examples.append((document[sent_idx], document[sent_idx + 1], 1))
        else:
            if len(documents) > 1:
                random_doc_idx = doc_idx
                while random_doc_idx == doc_idx:
                    random_doc_idx = random.randint(0, len(documents) - 1)
                random_document = documents[random_doc_idx]
            else:
                random_document = document
            random_sent_idx = random.randint(0, len(random_document) - 1)
            examples.append((document[sent_idx], random_document[random_sent_idx], 0))

    return examples[:num_examples]


class NSPHead:
    """Next Sentence Prediction classification head."""

    def __init__(self, hidden_size: int, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.W = torch.randn(hidden_size, 2, device=device) * 0.02
        self.b = torch.zeros(2, device=device)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(cls_hidden, self.W) + self.b


def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
