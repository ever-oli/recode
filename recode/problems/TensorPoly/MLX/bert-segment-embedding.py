import mlx.core as mx


class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """

    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.token_embeddings = mx.random.normal(shape=(vocab_size, hidden_size)) * 0.02
        self.position_embeddings = mx.random.normal(shape=(max_position, hidden_size)) * 0.02
        self.segment_embeddings = mx.random.normal(shape=(2, hidden_size)) * 0.02

    def forward(self, token_ids: mx.array, segment_ids: mx.array) -> mx.array:
        tok_emb = self.token_embeddings[token_ids]
        seq_len = token_ids.shape[1]
        positions = mx.arange(seq_len)
        pos_emb = self.position_embeddings[positions]
        seg_emb = self.segment_embeddings[segment_ids]
        return tok_emb + pos_emb + seg_emb
