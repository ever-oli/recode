from typing import List, Dict


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)

        current_id = len(special_tokens)
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        words = text.split()
        token_ids = []
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            token_ids.append(token_id)
        return token_ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            words.append(word)
        return " ".join(words)
