import copy
import logging
import os
from typing import Callable, Iterator, List, Union

import config
import numpy as np
import pandas as pd
import torch
from text import (
    DEFAULT_TOKEN,
    END_TOKEN,
    MASK_TOKEN,
    START_TOKEN,
    preprocess_text,
)
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def bpe_text_pipeline(
    input: Union[List[str], str],
    tokenizer: Callable[[str], List[str]],
) -> List[int]:
    """A utility to convert a list of strings or a string into tokens 
    defined by the trained tokenizer"""
    return (
        tokenizer.encode(preprocess_text(input)).ids
        if input.strip()
        else [tokenizer.token_to_id(DEFAULT_TOKEN)]
    )


def batch_iterator(
        df: pd.DataFrame,
        include_brand_cat=True
) -> Iterator[List[str]]:
    for _, row in df.iterrows():
        title = row["name"][0] if "name" in row and row["name"][0] else ""
        brand_name = (
            row["brand_name"][0] if "brand_name" in row and row["brand_name"][0] else ""  # noqa: E501
        )
        cat_name = (
            row["category_name"][0]
            if "category_name" in row and row["category_name"][0]
            else ""
        )
        yield preprocess_text(title) + " " + preprocess_text(
            brand_name
        ) + " " + preprocess_text(
            cat_name) if include_brand_cat else preprocess_text(
            title
        )


def train_tokenizer(df_train: pd.DataFrame) -> Tokenizer:
    tokenizer = Tokenizer(model=models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=config.BPE_VOCAB_LIMIT,
        special_tokens=[START_TOKEN, END_TOKEN, MASK_TOKEN, DEFAULT_TOKEN],
    )
    tokenizer.train_from_iterator(batch_iterator(df=df_train), trainer=trainer)
    logger.info(f"Constructed vocab of size: {config.BPE_VOCAB_LIMIT}")
    return tokenizer


def sequence_dataset(path, min_seq_len=10):
    data_files = os.listdir(path)
    df = pd.concat(
        [
            pd.read_parquet(
                os.path.join(path, file)
            ) for file in data_files
        ],
        ignore_index=True
    )
    df['seq_user_id'] = df['user_id'].astype(
        str) + "_" + df['sequence_id'].astype(str)
    df["category_name"] = df[config.CATEGORY_NAME_HIERARCHY].bfill(
        axis=1).iloc[:, 0]
    df["category_id"] = df[config.CATEGORY_ID_HIERARCHY].bfill(
        axis=1).iloc[:, 0]
    df = df.drop(config.CATEGORY_NAME_HIERARCHY +
                 config.CATEGORY_ID_HIERARCHY, axis=1)

    sequences = df.groupby('seq_user_id', as_index=False).agg(
        {
            'name': list,
            'category_name': list,
            'brand_name': list,
            'category_id': list,
            'brand_id': list,
            'item_id': list
        }
    )
    return sequences[sequences['name'].apply(len) >= min_seq_len]


def create_start_token_sequence(tokenizer, batch_size):
    start_dict = {"tokens": [], "offsets": [0]}
    for _ in range(batch_size):
        add_token(start_dict, tokenizer=tokenizer, token_type=START_TOKEN)
    start_dict["offsets"] = (
        torch.tensor(start_dict["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    return (
        torch.cat(start_dict["tokens"]).to(
            config.DEVICE), start_dict["offsets"]
    )


def add_token(collec, tokenizer: Tokenizer, token_type=START_TOKEN):
    collec["tokens"].append(torch.tensor([tokenizer.token_to_id(token_type)]))
    collec["offsets"].append(1)  # size of the token


def collate_batch_item(batch: torch.Tensor, tokenizer: Tokenizer):
    item_dict = {"tokens": [], "offsets": [0]}
    item_dict_y = {"tokens": [], "offsets": [0]}
    user_dict = {"tokens": [], "offsets": [0]}
    user_mask, item_mask = [], []

    for category, brand, title, _, _, _, _ in batch:
        # add start token to start of each sequence, see below for user
        add_token(item_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict["tokens"].append(tensor_feat)
            item_dict["offsets"].append(tensor_feat.size(0))
        # add end token to end of the item target sequence
        # add_token(item_dict, token_type=END_TOKEN)
        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict_y["tokens"].append(tensor_feat)
            item_dict_y["offsets"].append(tensor_feat.size(0))
        # add end token to end of the item target sequence
        add_token(item_dict_y, tokenizer=tokenizer, token_type=END_TOKEN)

        assert len(category) == len(brand) == len(
            title), "Batching not working"
        # add mask tokens to item sequence if needed
        item_mask.append(
            [True for _ in range(config.NUM_EVAL_SEQ + 1)]
        )  # 1 -> start or end token
        add_token(user_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        for ti, br, ca in zip(
            title[: -config.NUM_EVAL_SEQ],
            brand[: -config.NUM_EVAL_SEQ],
            category[: -config.NUM_EVAL_SEQ],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            user_dict["tokens"].append(tensor_feat)
            user_dict["offsets"].append(tensor_feat.size(0))
        add_token(user_dict, tokenizer=tokenizer, token_type=END_TOKEN)

        user_mask.append(
            [True for _ in range(len(category) - config.NUM_EVAL_SEQ + 2)]
        )  # 2 -> start + end tokens

        if len(category) - config.NUM_EVAL_SEQ < config.MODEL_SEQ_LEN:
            for _ in range(
                config.MODEL_SEQ_LEN - (len(category) - config.NUM_EVAL_SEQ)
            ):
                add_token(user_dict, tokenizer=tokenizer,
                          token_type=MASK_TOKEN)
                user_mask[-1].append(False)

    item_dict["offsets"] = (
        torch.tensor(item_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )
    item_dict_y["offsets"] = (
        torch.tensor(item_dict_y["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    user_dict["offsets"] = (
        torch.tensor(user_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )
    item_mask_y = copy.deepcopy(item_mask)

    return (
        (torch.cat(user_dict["tokens"]).to(
            config.DEVICE), user_dict["offsets"]),
        torch.from_numpy(np.array(user_mask)).to(config.DEVICE),
        (torch.cat(item_dict["tokens"]).to(
            config.DEVICE), item_dict["offsets"]),
        torch.from_numpy(np.array(item_mask)).to(config.DEVICE),
        (torch.cat(item_dict_y["tokens"]).to(
            config.DEVICE), item_dict_y["offsets"]),
        torch.from_numpy(np.array(item_mask_y)).to(config.DEVICE),
    )


def collate_batch_item_val(batch: torch.Tensor, tokenizer: Tokenizer):
    user_dict = {"tokens": [], "offsets": [0]}
    item_dict_y = {"tokens": [], "offsets": [0]}
    category_id_dict = {"tokens": []}
    brand_id_dict = {"tokens": []}
    item_id_dict = {"tokens": []}
    user_mask, item_mask = [], []

    for category, brand, title, category_id, brand_id, item_id, _ in batch:
        # add start token to the start of user sequence
        add_token(user_dict, tokenizer=tokenizer, token_type=START_TOKEN)
        # use only the first N - config.NUM_EVAL_SEQ
        for ti, br, ca in zip(
            title[: -config.NUM_EVAL_SEQ],
            brand[: -config.NUM_EVAL_SEQ],
            category[: -config.NUM_EVAL_SEQ],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            user_dict["tokens"].append(tensor_feat)
            user_dict["offsets"].append(tensor_feat.size(0))
        add_token(user_dict, tokenizer=tokenizer, token_type=END_TOKEN)

        assert len(category) == len(brand) == len(
            title), "Batching not working"
        user_mask.append(
            [True for _ in range(len(category) - config.NUM_EVAL_SEQ + 2)]
        )  # 2 -> start + end tokens

        if len(title) - config.NUM_EVAL_SEQ < config.MODEL_SEQ_LEN:
            for _ in range(config.MODEL_SEQ_LEN - (len(title) - config.NUM_EVAL_SEQ)):  # noqa: E501
                add_token(user_dict, tokenizer=tokenizer,
                          token_type=MASK_TOKEN)
                user_mask[-1].append(False)

        for ti, br, ca in zip(
            title[-config.NUM_EVAL_SEQ:],
            brand[-config.NUM_EVAL_SEQ:],
            category[-config.NUM_EVAL_SEQ:],
        ):
            concat_feat = preprocess_text(ti + " " + br + " " + ca)
            processed_feat = bpe_text_pipeline(concat_feat, tokenizer)
            assert processed_feat, "The text pipeline failed for features"
            if not processed_feat:
                processed_feat = [tokenizer.token_to_id(DEFAULT_TOKEN)]
            tensor_feat = torch.tensor(processed_feat, dtype=torch.long)
            item_dict_y["tokens"].append(tensor_feat)
            item_dict_y["offsets"].append(tensor_feat.size(0))

        item_mask.append([True for _ in range(config.NUM_EVAL_SEQ)])

        category_id_dict["tokens"].append(
            torch.tensor([category_id[-config.NUM_EVAL_SEQ:]],
                         dtype=torch.long)
        )
        brand_id_dict["tokens"].append(
            torch.tensor([brand_id[-config.NUM_EVAL_SEQ:]], dtype=torch.long)
        )

        item_id_dict["tokens"].append(
            torch.tensor(
                [item_id[-config.NUM_EVAL_SEQ:]],
                dtype=torch.long,
            )
        )

    item_dict_y["offsets"] = (
        torch.tensor(item_dict_y["offsets"][:-1]
                     ).cumsum(dim=0).to(config.DEVICE)
    )
    user_dict["offsets"] = (
        torch.tensor(user_dict["offsets"][:-1]).cumsum(dim=0).to(config.DEVICE)
    )

    return (
        (torch.cat(user_dict["tokens"]).to(
            config.DEVICE), user_dict["offsets"]),
        torch.from_numpy(np.array(user_mask)).to(config.DEVICE),
        (torch.cat(item_dict_y["tokens"]).to(
            config.DEVICE), item_dict_y["offsets"]),
        torch.from_numpy(np.array(item_mask)).to(config.DEVICE),
        torch.cat(category_id_dict["tokens"], dim=0).to(config.DEVICE),
        torch.cat(brand_id_dict["tokens"], dim=0).to(config.DEVICE),
        torch.cat(item_id_dict["tokens"], dim=0).to(config.DEVICE),
    )


class UserItemInteractionDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame):
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def _data_helper(self, events, tag):
        return [event if event else "" for event in events[tag]]

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = row['seq_user_id']
        category = self._data_helper(row, "category_name")
        brand = self._data_helper(row, "brand_name")
        title = self._data_helper(row, "name")
        category_id = [
            event if event else 0 for event in row["category_id"]
        ]
        brand_id = [event if event else 0 for event in row["brand_id"]]
        item_id = [event if event else 0 for event in row["item_id"]]
        return category, brand, title, category_id, brand_id, item_id, user_id
