import torch


class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """

    def __init__(self, vocab_size: int, max_position: int, hidden_size: int, device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.token_embeddings = torch.randn(vocab_size, hidden_size, device=device) * 0.02
        self.position_embeddings = torch.randn(max_position, hidden_size, device=device) * 0.02
        self.segment_embeddings = torch.randn(2, hidden_size, device=device) * 0.02

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        tok_emb = self.token_embeddings[token_ids]
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embeddings[positions]
        seg_emb = self.segment_embeddings[segment_ids]
        return tok_emb + pos_emb + seg_emb
