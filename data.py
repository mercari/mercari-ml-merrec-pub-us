"""
Partly dervied from
https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/master/utils.py
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model_def.ctr.inputs import (SparseFeat, VarLenSparseFeat,
                                  get_feature_names)


class Encode:
    def __init__(self):
        self.item_encoder = LabelEncoder()
    
    def fit_transform(self, df, key='product_id'):
        df[key] = self.item_encoder.fit_transform(df[key]) + 1
        return df
    
    def inverse_transform(self, df, key="product_id"):
        df[key] = self.item_encoder.inverse_transform(df[key]) - 1
        return df

class BertTrainDataset(Dataset):
    def __init__(self, sequences, max_len, mask_prob, pad_token, num_items, rng):
        self.sequences = sequences
        self.user_ids = sorted(self.sequences.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.pad_token = pad_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        seq = self.sequences[
            self.user_ids[index]
        ]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.pad_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [self.pad_token] * mask_len + tokens
        labels = [self.pad_token] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)


class MTLDataset(Dataset):
    def __init__(self, data, mtl_task_num):
        self.feature = data[0]
        self.mtl_task_num = mtl_task_num
        if mtl_task_num == 2:
            self.label1 = data[1]
            self.label2 = data[2]
        else:
            self.label = data[1]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        feature = self.feature[index]
        if self.mtl_task_num == 2:
            label1 = self.label1[index]
            label2 = self.label2[index]
            return feature, label1, label2
        else:
            label = self.label[index]
            return feature, label


class TrainDataset(Dataset):
    def __init__(self, sequences, max_len, pad_token):
        self.sequences = sequences
        self.user_ids = sorted(self.sequences.keys())
        self.max_len = max_len
        self.pad_token = pad_token
        print(len(self.user_ids))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        seq = self.sequences[
            self.user_ids[index]
        ]

        tokens = seq[:-1]
        labels = seq[1:]
        
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        x_len = len(tokens)
        y_len = len(labels)

        x_mask_len = self.max_len - x_len
        y_mask_len = self.max_len - y_len

        tokens = [self.pad_token] * x_mask_len + tokens
        labels = [self.pad_token] * y_mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)
    
class EvalDataset(Dataset):
    def __init__(self, sequences, target, max_len, pad_token, num_products):
        self.sequences = sequences
        self.user_ids = sorted(self.sequences.keys())
        self.target = target
        self.max_len = max_len
        self.pad_token = pad_token
        self.num_products = num_products + 1

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        seq = self.sequences[self.user_ids[index]][:-1]
        answer = self.target[self.user_ids[index]]
        answer = answer[-1:][0]

        labels = [0] * self.num_products
        labels[answer] = 1
        seq = seq + [self.pad_token]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [self.pad_token] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)

def sequence_dataset(path, min_seq_len=10, sample_prob=0.11):
    data_files = os.listdir(path)
    df = pd.concat(
        [
            pd.read_parquet(
                os.path.join(path, file)
            ) for file in data_files
        ], 
        ignore_index=True
    )
    df['seq_user_id'] = df['user_id'].astype(str) + "_" + df['sequence_id'].astype(str)
    product_count = len(set(df['product_id']))
    user_count = len(set(df['user_id']))
    print('Product Count: ', product_count)
    
    encoder = Encode()
    df = encoder.fit_transform(df=df, key='product_id')
    sequences = df.groupby('seq_user_id').product_id.apply(list).to_dict()
    del df
    filter_seq = {}
    for key in tqdm(sequences, desc="Filtering sequences"):
        if len(sequences[key]) >= min_seq_len and random.random() <= sample_prob: # keep roughly 10% of data
            filter_seq[key] = sequences[key]
    return filter_seq, product_count, user_count


def ctr_dataset(path=None):
    """
    Loader for CTR dataset.
    Derived from https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/43893d187e14c0b84e0f4d889477999ee831a3c9/utils.py#L416-L440
    """
    if not path:
        return
    df = pd.read_csv(path, usecols=[
        "user_id", "session_id", "product_id", "item_view",
        "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
        "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7",
    ])
    sparse_features = [
        "user_id", "session_id", "product_id",
        "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
        "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7"
    ]
    lbe = LabelEncoder()
    df['item_view'] = lbe.fit_transform(df['item_view'])

    for feat in tqdm(sparse_features, desc="[CTR] Creating feature columns"):
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    fixlen_feature_columns = [SparseFeat(feat, df[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
    train, test = train_test_split(df, test_size=0.1)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    print(f"CTR loaded Train and Test dataset sizes: train:{train.shape[0]}, test:{test.shape[0]}")
    return train, test, train_model_input, test_model_input, linear_feature_columns, dnn_feature_columns


def mtl_dataset(path=None, mtl_task_num=2):
    """
    Loader for MTL dataset.
    Derived from https://github.com/yuangh-x/2022-NIPS-Tenrec/blob/43893d187e14c0b84e0f4d889477999ee831a3c9/utils.py#L26-L68
    """
    if not path:
        return
    df = pd.read_csv(path, usecols=[
        "user_id", "session_id", "product_id", 
        "item_view", "item_like",
        "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
        "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7"
    ])
    if mtl_task_num == 2:
        label_columns = ['item_view', 'item_like']
        categorical_columns = [
            "user_id", "session_id", "product_id",
            "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
            "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7"
        ]
    elif mtl_task_num == 1:
        label_columns = ['item_view']
        categorical_columns = [
            "user_id", "session_id", "product_id",
            "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
            "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7"
        ]
    else:
        label_columns = ['item_like']
        categorical_columns = [
            "user_id", "session_id", "product_id",
            "c0_id", "c1_id", "c2_id", "brand_id", "size_id", "item_condition_id", "shipper_id", "color",
            "hist_1", "hist_2", "hist_3", "hist_4", "hist_5", "hist_6", "hist_7"
        ]
    user_columns = ["user_id", "session_id"]
    for col in tqdm(categorical_columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    new_columns = categorical_columns + label_columns
    df = df.reindex(columns=new_columns)

    user_feature_dict, item_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(df.columns), desc=f"MTL[{label_columns}] creating feature dicts"):
        if col not in label_columns:
            if col in user_columns:
                user_feature_dict[col] = (len(df[col].unique()), idx)
            else:
                item_feature_dict[col] = (len(df[col].unique()), idx)

    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]
    print(f"MTL[{label_columns}] loaded dataset sizes: train:{train_df.shape[0]}, val:{val_df.shape[0]}, test:{test_df.shape[0]}")
    return train_df, val_df, test_df, user_feature_dict, item_feature_dict


def train_val_test_split(sequences):
    assert sequences, "Sequences can't be None"
    tr_seq, val_seq, test_seq = {}, {}, {}
    for key, seq in tqdm(sequences.items()):
        tr_seq[key] = seq[:-2]
        val_seq[key] = seq[-2:-1]
        test_seq[key] = seq[-1:]

    return tr_seq, val_seq, test_seq

def get_data_loader(dataset, batch_size, is_parallel, is_train):
    if is_parallel:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=DistributedSampler(dataset)
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train, pin_memory=True)
    return dataloader
