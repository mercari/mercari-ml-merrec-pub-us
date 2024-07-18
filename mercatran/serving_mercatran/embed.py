import math

import torch
import torch.nn as nn


def create_user_target_mask(tgt_mask, device):
    attn_shape = (1, tgt_mask.size(-1), tgt_mask.size(-1))
    lower_triangle = (
        torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8).to(device)
    )
    lower_triangle = lower_triangle == 0
    return tgt_mask & lower_triangle


def create_item_encoder_mask(item_mask, device):
    seq_len = item_mask.size(-1)
    square_diagonal = torch.diag(torch.ones(
        seq_len)).type(torch.uint8).to(device)
    square_diagonal = square_diagonal == 1
    return item_mask & square_diagonal


def create_subsequent_mask(batch_size, seq_len, device):
    shape = (batch_size, seq_len, seq_len)
    subsequent_mask = (
        torch.triu(torch.ones(shape), diagonal=1).type(torch.uint8).to(device)
    )
    return subsequent_mask == 0


class UserEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_norm, padding_idx):
        super(UserEmbeddings, self).__init__()
        self.d_model = d_model
        self.user_embedding_bag = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            max_norm=max_norm,
            padding_idx=padding_idx,
        )

    def forward(self, x):
        return self.user_embedding_bag(x[0], x[1]) * math.sqrt(self.d_model)


class ItemEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_norm, padding_idx):
        super(ItemEmbeddings, self).__init__()
        self.d_model = d_model
        self.item_embedding_bag = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            max_norm=max_norm,
            padding_idx=padding_idx,
        )

    def forward(self, x):
        return self.item_embedding_bag(x[0], x[1]) * math.sqrt(self.d_model)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.custom_shape = args

    def forward(self, x):
        return x.view(self.custom_shape)


class PositionalEncoding(nn.Module):
    """Taken from: https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py#L748
    Computes sine and cosine positional embeddings"""  # noqa: E501

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
