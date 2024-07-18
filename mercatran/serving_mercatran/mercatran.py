import torch.nn as nn

import model_config
from base_model import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward,
)
from embed import ItemEmbeddings, PositionalEncoding, UserEmbeddings


class ThreeTower(nn.Module):
    def __init__(
        self,
        user_encoder=None,
        user_encoder_embed=None,
        user_decoder=None,
        user_decoder_embed=None,
        item_encoder=None,
        item_encoder_embed=None,
    ):
        super(ThreeTower, self).__init__()
        self.user_encoder = Encoder(  # user_encoder
            EncoderLayer(
                model_config.D_MODEL,
                MultiHeadedAttention(
                    model_config.NUM_HEADS, model_config.D_MODEL),
                PositionwiseFeedForward(
                    model_config.D_MODEL,
                    model_config.D_FF, model_config.DROPOUT
                ),
                model_config.DROPOUT,
            ),
            model_config.NUM_STACKS,
        )
        self.user_encoder_embed = nn.Sequential(  # user_encoder_embed
            UserEmbeddings(
                vocab_size=model_config.VOCAB_SIZE,
                d_model=model_config.D_MODEL,
                padding_idx=model_config.PADDING_IDX,
                max_norm=model_config.MAX_NORM,
            ),
            nn.Unflatten(0, (model_config.BATCH_SIZE, -1)),
            PositionalEncoding(model_config.D_MODEL, model_config.DROPOUT),
        )
        self.user_decoder = Decoder(  # user_decoder
            DecoderLayer(
                model_config.D_MODEL,
                MultiHeadedAttention(
                    model_config.NUM_HEADS, model_config.D_MODEL),
                MultiHeadedAttention(
                    model_config.NUM_HEADS, model_config.D_MODEL),
                PositionwiseFeedForward(
                    model_config.D_MODEL,
                    model_config.D_FF, model_config.DROPOUT
                ),
                model_config.DROPOUT,
            ),
            model_config.NUM_STACKS,
        )
        self.user_decoder_embed = nn.Sequential(  # user_decoder_embed
            ItemEmbeddings(
                vocab_size=model_config.VOCAB_SIZE,
                d_model=model_config.D_MODEL,
                padding_idx=model_config.PADDING_IDX,
                max_norm=model_config.MAX_NORM,
            ),
            nn.Unflatten(0, (model_config.BATCH_SIZE, -1)),
            PositionalEncoding(model_config.D_MODEL, model_config.DROPOUT),
        )
        self.item_encoder = Encoder(  # item_encoder
            EncoderLayer(
                model_config.D_MODEL,
                MultiHeadedAttention(
                    model_config.NUM_HEADS, model_config.D_MODEL),
                PositionwiseFeedForward(
                    model_config.D_MODEL,
                    model_config.D_FF, model_config.DROPOUT
                ),
                model_config.DROPOUT,
            ),
            model_config.NUM_STACKS,
        )
        self.item_encoder_embed = nn.Sequential(  # item_encoder_embed
            ItemEmbeddings(
                vocab_size=model_config.VOCAB_SIZE,
                d_model=model_config.D_MODEL,
                padding_idx=model_config.PADDING_IDX,
                max_norm=model_config.MAX_NORM,
            ),
            nn.Unflatten(0, (model_config.BATCH_SIZE, -1)),
        )

    def forward(self, user, user_mask, item, item_mask, item_y, item_mask_y):
        # Note: We don't use this method right now
        user_enc_out = self.user_encoder(
            self.user_encoder_embed(user), user_mask)
        user_dec_out = self.user_decoder(
            self.user_decoder_embed(item), user_enc_out, user_mask, item_mask
        )
        item_enc_out = self.item_encoder(
            self.item_encoder_embed(item_y), item_mask_y)
        return user_dec_out, item_enc_out
