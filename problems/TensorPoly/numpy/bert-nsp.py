import numpy as np
from typing import List, Tuple
import random


def create_nsp_examples(documents: List[List[str]], num_examples: int, seed: int = None) -> List[Tuple[str, str, int]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

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

    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        return np.dot(cls_hidden, self.W) + self.b


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
