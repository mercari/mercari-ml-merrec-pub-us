from typing import Tuple

import numpy as np
import torch

import mercatran_handler_base
import model_config
from embed import create_subsequent_mask


class MercatranHandlerUser(mercatran_handler_base.MercatranHandlerBase):
    def _collate_user(
        self, batch: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]:
        """Data formatter that can tokenize item title, brand and category,"""
        user_dict = {"tokens": [], "offsets": [0]}
        user_mask = []
        padded_slots = 0

        for user in batch.get("user"):
            # add start token to start each sequence
            self._add_token(user_dict, token_type=model_config.START_TOKEN)
            title, brand, category = (
                user.get("title"),
                user.get("brand"),
                user.get("category"),
            )
            assert (
                len(title) == len(brand) == len(category)
            ), "Individual features are of unequal length"
            assert (
                0 < len(title) <= model_config.MODEL_SEQ_LEN
            ), f"Sequence length should be b/w 1 and {model_config.MODEL_SEQ_LEN}"  # noqa E501
            for ti, br, ca in zip(title, brand, category):
                if not ti:
                    ti = ""
                if not br:
                    br = ""
                if not ca:
                    ca = ""
                concat_feat = self._preprocess_text(ti + " " + br + " " + ca)
                processed_feat = self.bpe_text_pipeline(
                    input=concat_feat, tokenizer=self.source_vocab
                )
                if not processed_feat:
                    processed_feat = [
                        self.source_vocab.token_to_id(
                            model_config.DEFAULT_TOKEN)
                    ]
                assert (
                    processed_feat
                ), "The text pipeline failed for the user input features"
                tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
                user_dict["tokens"].append(tensor_feat)
                user_dict["offsets"].append(tensor_feat.size(0))
            # add end token to the end of user sequence
            self._add_token(user_dict, token_type=model_config.END_TOKEN)
            user_mask.append(
                # 2 for start and end tokens
                [True for _ in range(len(title) + 2)]
            )
            if len(title) < model_config.MODEL_SEQ_LEN:
                for _ in range(model_config.MODEL_SEQ_LEN - len(title)):
                    self._add_token(
                        user_dict, token_type=model_config.MASK_TOKEN)
                    user_mask[-1].append(False)

        if len(user_mask) < model_config.BATCH_SIZE:
            padded_slots = model_config.BATCH_SIZE - len(user_mask)
            for _ in range(padded_slots):
                self._add_row(user_dict, token_type=model_config.MASK_TOKEN)
                user_mask.append(
                    [False for _ in range(model_config.MODEL_SEQ_LEN + 2)])

        user_dict["offsets"] = (
            torch.tensor(user_dict["offsets"][:-1]
                         ).cumsum(dim=0).to(self.device)
        )

        return (
            (torch.cat(user_dict["tokens"]).to(
                self.device), user_dict["offsets"]),
            torch.from_numpy(np.array(user_mask)).to(self.device),
            padded_slots,
        )

    def _create_start_token_sequence(self, batch_size=model_config.BATCH_SIZE):
        start_dict = {"tokens": [], "offsets": [0]}
        for _ in range(batch_size):
            self._add_token(start_dict, model_config.START_TOKEN)
        start_dict["offsets"] = (
            torch.tensor(start_dict["offsets"][:-1]
                         ).cumsum(dim=0).to(self.device)
        )
        return (
            torch.cat(start_dict["tokens"]).to(
                self.device), start_dict["offsets"]
        )

    def preprocess(self, data):
        """Preprocesses the data before calling the collation function.
        Ensure the input format is consistent with the sample format
        provided in the examples."""
        data = data[0]
        return self._collate_user(data)

    def inference(self, data, *args, **kwargs):
        user, user_mask, padded_slots = data
        results = []
        with torch.no_grad():
            self.model.eval()
            user_enc_out = self.model.user_encoder(
                self.model.user_encoder_embed(user), user_mask.unsqueeze(-2)
            )
            dec_embed = self.model.user_decoder_embed(
                self._create_start_token_sequence(
                    batch_size=model_config.BATCH_SIZE)
            )
            for _ in range(model_config.NUM_PRED_STEPS):
                user_dec_out = self.model.user_decoder(
                    dec_embed,
                    user_enc_out,
                    user_mask.unsqueeze(-2),
                    create_subsequent_mask(
                        model_config.BATCH_SIZE, dec_embed.size(1), self.device
                    ),
                )
                # we track number of input requests and return the non-padded
                # results == model_config.BATCH_SIZE - padded_slots
                results.append(
                    user_dec_out[: model_config.BATCH_SIZE -
                                 padded_slots, -1, :]
                    .cpu()
                    .numpy()
                )
                dec_embed = torch.cat(
                    (dec_embed, user_dec_out[:, -1, :].unsqueeze(1)), dim=1
                )
        return results
