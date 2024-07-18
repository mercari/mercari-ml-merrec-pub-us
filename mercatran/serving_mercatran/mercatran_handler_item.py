from typing import Tuple

import numpy as np
import torch

import mercatran_handler_base
import model_config
from embed import create_item_encoder_mask


class MercatranHandlerItem(mercatran_handler_base.MercatranHandlerBase):
    def _collate_item(
        self, batch: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, int]:
        """Mini batch data formatter that can tokenize item details such as
        item title, brand and category. Note: Each sequence can only have 1 
        event, corresponding to one item"""
        item_dict = {"tokens": [], "offsets": [0]}
        item_mask = []
        padded_slots = 0

        for item in batch.get("item"):
            title, brand, category = (
                item.get("title"),
                item.get("brand"),
                item.get("category"),
            )
            assert (
                len(title) == len(brand) == len(category) == 1
            ), "Individual features need to be equal to 1"
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
                ), "The text pipeline failed for the item input features"
                tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
                item_dict["tokens"].append(tensor_feat)
                item_dict["offsets"].append(tensor_feat.size(0))
            # Note: We don't append START or END tokens to the input
            item_mask.append([True for _ in range(len(title))])

        if len(item_mask) < model_config.BATCH_SIZE:
            padded_slots = model_config.BATCH_SIZE - len(item_mask)
            for _ in range(padded_slots):
                # since each row will have 1 item, we add one token per row
                self._add_row(
                    item_dict, num_tokens=1, token_type=model_config.MASK_TOKEN
                )
                item_mask.append([False for _ in range(len(title))])

        item_dict["offsets"] = (
            torch.tensor(item_dict["offsets"][:-1]
                         ).cumsum(dim=0).to(self.device)
        )

        return (
            (torch.cat(item_dict["tokens"]).to(
                self.device), item_dict["offsets"]),
            torch.from_numpy(np.array(item_mask)).to(self.device),
            padded_slots,
        )

    def preprocess(self, data):
        """Preprocesses the data before calling the collation function.
        Ensure the input format is consistent with the sample
        format provided in the examples."""
        data = data[0]
        return self._collate_item(data)

    def inference(self, data, *args, **kwargs):
        item, item_mask, padded_slots = data
        results = []
        with torch.no_grad():
            self.model.eval()
            item_embedding = self.model.item_encoder(
                self.model.item_encoder_embed(item),
                create_item_encoder_mask(item_mask.unsqueeze(-2), self.device),
            )
            # we track the number of input request and return the non-padded
            # results == model_config.BATCH_SIZE - padded_slots
            results.append(
                item_embedding[: model_config.BATCH_SIZE - padded_slots, :, :]
                .cpu()
                .numpy()
            )
        return results
